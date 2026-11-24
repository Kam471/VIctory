//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and denormal
//! operands of `v_mul_legacy_f32`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// SOPC `op src0, src1`.
const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32 & 0x7F) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// SOPP `op simm16`.
const fn sopp(op: u8, simm16: i16) -> u32 {
    0xBF80_0000 | (op as u32 & 0x7F) << 16 | simm16 as u16 as u32
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3B `op vdst, sdst, src0, src1, src2`.
const fn vop3b(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | (sdst as u32 & 0x7F) << 8 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// GLOBAL `global_load_dword vdst, vaddr, saddr` (`saddr = 125` for null).
const fn global_load_dword(vdst: u8, vaddr: u8, saddr: u8, offset: i16) -> [u32; 2] {
    [
        0x37 << 26 | 0x0C << 18 | 2 << 14 | (offset as u32 & 0xFFF),
        (vdst as u32) << 24 | (saddr as u32 & 0x7F) << 16 | vaddr as u32,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    try_build_valu_cs(body).expect("lower")
}

/// The synthetic program [`build_valu_cs`] and [`try_build_valu_cs`] lower.
fn valu_program(body: &[u32]) -> NativeProgram {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    }
}

/// [`build_valu_cs`], with the lowering's refusal as an error.
fn try_build_valu_cs(body: &[u32]) -> Result<HostShader, String> {
    let native = valu_program(body);

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        psv: None,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        options: Default::default(),
        host_waves: None,
    })
    .map_err(|e| format!("{e:?}"))?;

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        stride_table: lowered.stride_table,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    Ok(shader)
}

/// Emits `native` as the GPU layer does, from a container carrying `parts`.
fn lower_test_shader(native: &NativeProgram, parts: &[(crate::FourCC, &[u8])]) -> HostShader {
    let bytes = crate::dxbc::build(parts);
    let container = crate::Container::parse(&bytes).expect("container");
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let shader =
        super::emit(&container, native, &p, &st, &io, &res).expect("emit");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }
    shader
}

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    run_with_views(shader, cases, &[View::RawUav(1)], check);
}

fn run_with_views(
    shader: &HostShader,
    cases: &[[u32; 4]],
    views: &[View<'_>],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn data_only_structured_srv_reads_its_bound_record_count() {
    // The second heap slot contains an SRV. Its only use is a read of dword 2.
    let mut load = s_load_dwordx4(12, 4);
    load[1] |= 32;
    let mut body = load.to_vec();
    body.push(vop1(0x001, 0, sgpr(14)));
    let mut native = valu_program(&body);
    native.used.push(UsedResource {
        kind: ResourceType::Srv,
        table_offset_dwords: 8,
        lower_bound: 109,
        stride: 16,
        ..native.used[0]
    });
    use super::container::{Binding, Psv};
    let binding = Binding {
        kind: 5,
        space: 1,
        register: 109,
        dxil_kind: 12,
    };
    let declarations = |bindings: &[Binding]| {
        super::container::psv(&Psv {
            stage: 5,
            flag: false,
            inputs: &[],
            outputs: &[],
            prims: &[],
            mesh: None,
            bindings,
            threads: [1, 1, 1],
            wave_size: None,
        })
    };
    let psv = declarations(&[binding]);
    let alias = Binding {
        kind: 3,
        space: 2,
        dxil_kind: 10,
        ..binding
    };
    assert!(!super::declares_structured_buffer(
        &declarations(&[binding, alias]),
        109,
        false
    ));
    assert!(!super::declares_structured_buffer(&psv, 108, false));
    let shader = lower_test_shader(&native, &[(crate::FourCC::PSV0, &psv)]);
    for n in [3, 11] {
        let words = vec![0; n * 4];
        let views = [
            View::RawUav(1),
            View::Buffer {
                data: &words,
                stride: super::host::Stride::Structured(16),
            },
        ];
        run_with_views(&shader, &[[0; 4]], &views, |device, _, _, out| {
            assert_eq!(out, n as u32, "{device} structured buffer record count");
        });
    }
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn root_uav_validates_and_dispatches_on_gpu_and_warp() {
    let s_load_dword = |dst: u8, base: u8, byte_off: u32| -> [u32; 2] {
        [
            0x3D << 26 | (dst as u32) << 6 | (base as u32 >> 1),
            125 << 25 | (byte_off & 0x1F_FFFF),
        ]
    };
    let buffer_load_dword = |vdst: u8, srsrc: u8, byte_off: u16| -> [u32; 2] {
        [
            0x38 << 26 | 0x0C << 18 | (byte_off as u32 & 0xFFF),
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8,
        ]
    };
    let buffer_atomic_add_rtn = |vdata: u8, srsrc: u8, byte_off: u16| -> [u32; 2] {
        [
            0x38 << 26 | 0x32 << 18 | (1 << 14) | (byte_off as u32 & 0xFFF),
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
        ]
    };
    let global_store_dword = |vaddr: u8, vdata: u8, saddr: u8, byte_off: u16| -> [u32; 2] {
        [
            0x37 << 26 | 0x1C << 18 | (2 << 14) | (byte_off as u32 & 0xFFF),
            (saddr as u32) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    };

    let mut words = Vec::new();
    // s[0:1] = Root CBV (slot 0), s[2:3] = Root SRV (slot 1), s[4:5] = Root UAV (slot 2).
    words.extend_from_slice(&s_load_dword(6, 0, 8)); // s6 = cbv[2]
    words.extend_from_slice(&s_load_dword(7, 2, 4)); // s7 = srv[1]
    // Build a raw buffer descriptor around Root UAV s[4:5] in s[8:11].
    words.push(s_mov_b32(8, 4));
    words.push(s_mov_b32(9, 5));
    words.push(s_mov_b32(10, 0xC1)); // -1
    words.extend_from_slice(&[s_mov_b32(11, 0xFF), 0x3101_4FAC]);
    // uav[0] = s6 via global_store_dword.
    words.push(vop1(0x001, 0, 0x80)); // v0 = 0
    words.push(vop1(0x001, 1, sgpr(6))); // v1 = s6
    words.extend_from_slice(&global_store_dword(0, 1, 4, 0));
    // uav[1] = s7 via buffer_store_dword, then atomic_add 17 (old value -> v3).
    words.push(vop1(0x001, 2, sgpr(7))); // v2 = s7
    words.extend_from_slice(&buffer_store_dword(2, 8, 4));
    words.push(vop1(0x001, 3, 128 + 17)); // v3 = 17
    words.extend_from_slice(&buffer_atomic_add_rtn(3, 8, 4));
    // v4 = uav[0] via buffer_load_dword; uav[2] = v3 ^ v4.
    words.extend_from_slice(&buffer_load_dword(4, 8, 0));
    let vop2 = |op: u8, vdst: u8, src0: u16, vsrc1: u8| -> u32 {
        (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    };
    words.push(vop2(0x1D, 5, vgpr(3), 4)); // v_xor_b32 v5, v3, v4
    words.extend_from_slice(&buffer_store_dword(5, 8, 8));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 1,
                user_data: 2,
                slot: 1,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 2,
                user_data: 4,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 4,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 6 << 1,
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let container = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&container).expect("container");
    let shader = super::emit(&c, &native, &p, &st, &io, &res).expect("emit");

    assert_eq!(shader.root_cbvs, vec![0]);
    assert_eq!(shader.root_srvs, vec![1]);
    assert_eq!(shader.root_uavs, vec![2]);

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // Root parameter 0 is user data; sorted root views follow at 1 (CBV 0), 2 (SRV 1), 3 (UAV 2).
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(vec![0; 4])),
                        (
                            1,
                            Root::View {
                                srv: false,
                                data: vec![0x1111_1111, 0x2222_2222, 0xCAFE_BEEF, 0],
                            },
                        ),
                        (
                            2,
                            Root::View {
                                srv: true,
                                data: vec![0, 0x1000, 0, 0],
                            },
                        ),
                        (3, Root::Uav(4)),
                    ],
                    views: &[],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(
            &out[0][..3],
            &[0xCAFE_BEEF, 0x1011, 0x1000 ^ 0xCAFE_BEEF],
            "{label}: unexpected Root UAV readback"
        );
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn root_uav_graphics_stages_validate_and_build_pso() {
    use super::bitcode::{Attr, Term};
    use super::container::SigElement;
    use super::shader::Shader;

    let make_stage = |pixel: bool, slots: &[u8]| -> HostShader {
        let mut s = Shader::new(pixel);
        let b = s.entry;
        let iu = s.m.undef(s.t.i32);
        for (k, &slot) in slots.iter().enumerate() {
            let h = s.root_uav_handle(slot);
            let args = [
                h,
                s.m.i32((k * 4) as u32),
                iu,
                s.m.i32(0x1234_0000 + u32::from(slot)),
                iu,
                iu,
                iu,
                s.m.i8(1),
                s.m.i32(4),
            ];
            s.op(
                b,
                "rawBufferStore.i32",
                Attr::NoUnwind,
                s.t.void,
                140,
                &args,
            );
        }
        let out = SigElement {
            name: if pixel { "SV_Target" } else { "SV_Position" }.into(),
            index: 0,
            kind: if pixel { 16 } else { 3 },
            comp: 9,
            interp: if pixel { 0 } else { 4 },
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        for col in 0..4u8 {
            let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), s.m.f32(0)];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.outputs.push(out);
        if pixel {
            s.inputs.push(SigElement {
                name: "SV_Position".into(),
                index: 0,
                kind: 3,
                comp: 9,
                interp: 4,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 0,
                half: false,
            });
        }
        s.m.terminate(b, Term::Ret);
        HostShader {
            stage: if pixel { HwStage::Ps } else { HwStage::Vs },
            dxil: s.finish(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: slots.to_vec(),
            stride_table: false,
        }
    };

    let vs = make_stage(false, &[2]);
    let ps = make_stage(true, &[2, 3]);

    // Vertex shaders writing UAVs must stamp SFI0 UAVsAtEveryStage (0x4).
    let vs_c = crate::Container::parse(&vs.dxil).unwrap();
    let sfi0 = u64::from_le_bytes(
        vs_c.part(crate::FourCC::SFI0)
            .unwrap()
            .data
            .try_into()
            .unwrap(),
    );
    assert_ne!(
        sfi0 & 0x4,
        0,
        "VS with Root UAV must set SFI0 UAVsAtEveryStage"
    );

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        for (name, sh) in [("vs", &vs), ("ps", &ps)] {
            let mut copy = sh.dxil.clone();
            let ours = copy[4..20].to_vec();
            v.validate(&mut copy)
                .unwrap_or_else(|e| panic!("{name}: DXIL validation failed: {e}"));
            assert_eq!(copy[4..20], ours[..], "{name}: container hash mismatch");
        }
    }

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        host.pipeline(&vs, &ps).unwrap_or_else(|e| {
            panic!("{label}: graphics pipeline with shared Root UAV failed: {e}")
        });
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

fn build_root_srv_cs(body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 for VALU address expressions.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1, // 8 user SGPRs: s0..s7
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        psv: None,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        options: Default::default(),
        host_waves: None,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        stride_table: lowered.stride_table,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// Runs a Root SRV compute shader on available SM 6.6 devices (hardware GPU and WARP).
fn run_root_srv_cases(
    shader: &HostShader,
    srv_data: &[u32],
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: srv_data.to_vec(),
                                },
                            ),
                        ],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn root_srv_s_lshl2_add_u32_with_root_constant_dword0() {
    // s_lshl2_add_u32 s12, s0, s6; s_add_u32 s12, s12, s1; s_addc_u32 s13, s7, 0;
    // v_mov_b32 v5, 0; global_load_dword v0, v5, s[12:13]
    let shader = build_root_srv_cs(&[
        sop2(0x2F, 12, 0, 6),
        sop2(0x00, 12, 12, 1),
        sop2(0x04, 13, 7, 0x80),
        vop1(0x001, 5, 0x80),
        global_load_dword(0, 5, 12, 0)[0],
        global_load_dword(0, 5, 12, 0)[1],
    ]);
    let srv_data: Vec<u32> = (0..16).map(|i| 0x1000_0000 + i * 0x0101_0101).collect();
    let cases = [[0, 0, 0, 0], [3, 0, 0, 0], [5, 8, 0, 0], [11, 16, 0, 0]];
    run_root_srv_cases(&shader, &srv_data, &cases, |backend, _, inp, got| {
        let want = srv_data[(inp[0] + inp[1] / 4) as usize];
        assert_eq!(
            got, want,
            "{backend}: s_lshl2_add_u32 index {} + {}B = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn root_srv_two_term_v_add_co_and_v_lshl_add() {
    // v_add_co_u32 v5, vcc, s6, v1
    // v_add_co_ci_u32 v6, vcc, s7, 0, vcc (VOP3B op 0x128)
    // v_lshl_add_u32 v5, v2, 2, v5
    // global_load_dword v0, v[5:6], off
    let add_co = vop3b(0x30F, 5, 106, sgpr(6), vgpr(1), 0);
    let add_ci = vop3b(0x128, 6, 106, sgpr(7), 0x80, sgpr(106));
    let lshl_add = vop3(0x346, 5, vgpr(2), 0x82, vgpr(5));
    let load = global_load_dword(0, 5, 125, 0);
    let shader = build_root_srv_cs(&[
        add_co[0],
        add_co[1],
        add_ci[0],
        add_ci[1],
        lshl_add[0],
        lshl_add[1],
        load[0],
        load[1],
    ]);
    let srv_data: Vec<u32> = (0..24).map(|i| 0xA000_0000 ^ (i * 0x0011_2233)).collect();
    let cases = [[0, 0, 0, 0], [8, 1, 0, 0], [4, 5, 0, 0], [16, 7, 0, 0]];
    run_root_srv_cases(&shader, &srv_data, &cases, |backend, _, inp, got| {
        let idx = (inp[0] / 4 + inp[1]) as usize;
        let want = srv_data[idx];
        assert_eq!(
            got, want,
            "{backend}: two-term v_add_co + v_lshl_add (off={}, idx={}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn root_srv_loop_carried_pointer_induction_phi() {
    // s12, s13 = root_srv + (s0 << 2); s14 = 0; v0 = 0; v5 = 0;
    // loop:
    //   global_load_dword v6, v5, s[12:13]   (2 dwords)
    //   v_add_nc_u32 v0, v0, v6              (1 dword)
    //   s_add_u32 s12, s12, 4                (1 dword)
    //   s_addc_u32 s13, s13, 0               (1 dword)
    //   s_add_u32 s14, s14, 1                (1 dword)
    //   s_cmp_lt_u32 s14, 4                  (1 dword)
    //   s_cbranch_scc1 -8                    (jumps back 8 dwords -> start of loop)
    let load = global_load_dword(6, 5, 12, 0);
    let shader = build_root_srv_cs(&[
        sop2(0x2F, 12, 0, 6),      // s_lshl2_add_u32 s12, s0, s6
        s_mov_b32(13, 7),          // s_mov_b32 s13, s7
        s_mov_b32(14, 0x80),       // s_mov_b32 s14, 0
        vop1(0x001, 0, 0x80),      // v_mov_b32 v0, 0
        vop1(0x001, 5, 0x80),      // v_mov_b32 v5, 0
        load[0],                   // loop start (-7 before branch)
        load[1],                   // (-6)
        vop2(0x25, 0, vgpr(0), 6), // v_add_nc_u32 v0, v0, v6 (-5)
        sop2(0x00, 12, 12, 0x84),  // s_add_u32 s12, s12, 4 (-4)
        sop2(0x04, 13, 13, 0x80),  // s_addc_u32 s13, s13, 0 (-3)
        sop2(0x00, 14, 14, 0x81),  // s_add_u32 s14, s14, 1 (-2)
        sopc(0x0A, 14, 0x84),      // s_cmp_lt_u32 s14, 4 (-1)
        sopp(0x05, -8),            // s_cbranch_scc1 loop
    ]);
    let srv_data: Vec<u32> = (0..16).map(|i| 100 + i * 17).collect();
    let cases = [[0, 0, 0, 0], [2, 0, 0, 0], [5, 0, 0, 0], [11, 0, 0, 0]];
    run_root_srv_cases(&shader, &srv_data, &cases, |backend, _, inp, got| {
        let start = inp[0] as usize;
        let want: u32 = srv_data[start..start + 4]
            .iter()
            .fold(0u32, |a, &x| a.wrapping_add(x));
        assert_eq!(
            got, want,
            "{backend}: loop phi sum starting at {start} = {got}, want {want}",
        );
    });
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}



fn build_wave32_cs(body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    // v2 = v0 * 4 (per-lane byte offset into the UAV), store v1 at out[v0].
    words.push((0x01A << 25) | (2 << 17) | 0x82); // v_lshlrev_b32 v2, 2, v0
    words.extend_from_slice(&[
        0x38 << 26 | 0x1C << 18 | (1 << 12), // buffer_store_dword ... offen
        0x80 << 24 | ((8 >> 2) << 16) | (1 << 8) | 2,
    ]);
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [32, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        psv: None,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        options: Default::default(),
        host_waves: None,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        stride_table: lowered.stride_table,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

fn dispatch_wave32(
    name: &str,
    shader: &HostShader,
    fixed_wave: bool,
    cases: &[([u32; 4], Vec<u32>)],
) {
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("{name}: no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            // WARP does not accept every fixed WaveSize requirement.
            Err(e) if warp && fixed_wave => {
                eprintln!("{name}: skipped on {label} ({e})");
                continue;
            }
            Err(e) => panic!("{name} ({label}): compute_pipeline failed: {e}"),
        };
        for (i, (inputs, want)) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(32)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{name} ({label}): dispatch failed: {e}"));
            assert_eq!(&out[0], want, "{name} ({label}): case {i} ({inputs:?})");
        }
        ran += 1;
    }
    assert!(
        ran > 0,
        "{name}: neither hardware nor WARP SM 6.6 device available"
    );
}

#[test]
fn lane_mask_phi_joins_execute_correctly() {
    // Branch join: s0 == 0 sets s2 = 0; otherwise s2 = (v0 < s1).
    let branch = build_wave32_cs(&[
        0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
        0xBF84_0002, // 4: s_cbranch_scc0 -> 16 (otherwise s2 = (v0 < s1))
        0xBE82_0380, // 8: s_mov_b32 s2, 0
        0xBF82_0002, // 12: s_branch -> 24
        0xD4C4_0002,
        u32::from(sgpr(1)) | (u32::from(vgpr(0)) << 9), // 16: v_cmp_gt_u32 s2, s1, v0 (v0 < s1)
        0xD501_0001,
        (0x8A) | (0x94 << 9) | (2 << 18), // 24: v_cndmask_b32 v1, 10, 20, s2
    ]);
    dispatch_wave32(
        "branch_zero_join",
        &branch,
        false,
        &[
            ([0, 16, 0, 0], vec![10; 32]),
            (
                [1, 16, 0, 0],
                (0..32).map(|t| if t < 16 { 20 } else { 10 }).collect(),
            ),
        ],
    );

    // Branch join with a partial constant mask: s0 == 0 sets s2 = 0xF; otherwise s2 = (v0 < s1).
    let partial_constant = build_wave32_cs(&[
        0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
        0xBF84_0002, // 4: s_cbranch_scc0 -> 16 (otherwise s2 = (v0 < s1))
        0xBE82_038F, // 8: s_mov_b32 s2, 15
        0xBF82_0002, // 12: s_branch -> 24
        0xD4C4_0002,
        u32::from(sgpr(1)) | (u32::from(vgpr(0)) << 9), // 16: v_cmp_gt_u32 s2, s1, v0 (v0 < s1)
        0xD501_0001,
        (0x8A) | (0x94 << 9) | (2 << 18), // 24: v_cndmask_b32 v1, 10, 20, s2
    ]);
    dispatch_wave32(
        "partial_constant_join",
        &partial_constant,
        true,
        &[
            (
                [0, 16, 0, 0],
                (0..32).map(|t| if t < 4 { 20 } else { 10 }).collect(),
            ),
            (
                [1, 16, 0, 0],
                (0..32).map(|t| if t < 16 { 20 } else { 10 }).collect(),
            ),
        ],
    );

    // Loop accumulator from 0: iterates i = 0..s0, reading phi s2 inside the
    // loop, accumulating `s2 |= (v0 == i)`, then reading final s2 after exit.
    // v1 = (inside last iter: 20 if v0 < s0 - 1 else 10) + (after loop: 30 if v0 < s0 else 10).
    let loop_shader = build_wave32_cs(&[
        0xBE82_0380, // 0: s_mov_b32 s2, 0
        0xBE86_0380, // 4: s_mov_b32 s6, 0
        0xD501_0001,
        (0x8A) | (0x94 << 9) | (2 << 18), // 8: v_cndmask_b32 v1, 10, 20, s2
        0x7D84_0006,                      // 16: v_cmp_eq_u32 vcc, s6, v0
        0x8802_026A,                      // 20: s_or_b32 s2, vcc_lo, s2
        0x8006_8106,                      // 24: s_add_u32 s6, s6, 1
        0xBF0A_0006,                      // 28: s_cmp_lt_u32 s6, s0
        0xBF85_FFF9,                      // 32: s_cbranch_scc1 -> 8
        0xD501_0003,
        (0x8A) | (0x9E << 9) | (2 << 18), // 36: v_cndmask_b32 v3, 10, 30, s2
        (0x025 << 25) | (1 << 17) | (3 << 9) | 257, // 44: v_add_nc_u32 v1, v1, v3
    ]);
    dispatch_wave32(
        "loop_accumulator",
        &loop_shader,
        false,
        &[(
            [5, 0, 0, 0],
            (0..32)
                .map(|t| {
                    if t < 4 {
                        50
                    } else if t == 4 {
                        40
                    } else {
                        20
                    }
                })
                .collect(),
        )],
    );
}

#[test]
fn v_mul_legacy_f32_treats_denormals_as_zero() {
    // v_mul_legacy_f32 v0, v1, v2 (VOP2 opcode 0x007 in its VOP3 encoding)
    let shader = build_valu_cs(&vop3(0x107, 0, vgpr(1), vgpr(2), vgpr(0)));
    let f = f32::to_bits;
    let (denorm, neg_denorm, min_normal) = (0x0000_0001u32, 0x8040_0000u32, 0x0080_0000u32);

    let cases: &[(u32, u32, u32)] = &[
        // Zero or denormal times anything, non-finite included, is +0.
        (f(0.0), f(f32::INFINITY), f(0.0)),
        (f(-0.0), f(f32::NAN), f(0.0)),
        (f(f32::NAN), f(0.0), f(0.0)),
        (denorm, f(f32::INFINITY), f(0.0)),
        (f(f32::NAN), neg_denorm, f(0.0)),
        (denorm, f(2.0f32.powi(100)), f(0.0)),
        // The smallest normal is not zero.
        (min_normal, f(2.0), f(f32::MIN_POSITIVE * 2.0)),
        // Ordinary products.
        (f(1.5), f(-4.0), f(-6.0)),
        (f(3.0), f(0.5), f(1.5)),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(a, b, _)| [a, b, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: v_mul_legacy_f32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

/// MTBUF `op vdata, v5, s[8:11], 0 format:fmt offset:16` (no offen/idxen;
/// `op` 4..7 stores x..xyzw, 12..15 their d16 forms).
const fn mtbuf_store(op: u8, fmt: u8, vdata: u8) -> [u32; 2] {
    [
        0x3A << 26 | (fmt as u32) << 19 | (op as u32 & 7) << 16 | 16,
        0x80 << 24 | (op as u32 >> 3 & 1) << 21 | (8 >> 2) << 16 | (vdata as u32) << 8 | 5,
    ]
}

/// Runs `shader` once with root constants `inputs` on every available
/// SM 6.6 host and returns the first `dwords` of its raw UAV, per device.
fn run_dwords(shader: &HostShader, inputs: [u32; 4], dwords: u32) -> Vec<(&'static str, Vec<u32>)> {
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) => inputs[u.dword as usize],
                (RootType::Table, 1) => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let r = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(dwords)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, r[0].clone()));
    }
    assert!(
        !out.is_empty(),
        "neither hardware nor WARP SM 6.6 device available"
    );
    out
}

#[test]
fn tbuffer_store_format_writes_the_formats_layout() {
    let f = |x: f32| x.to_bits();
    // (what, op, format, data in v1..v4, dwords 4..8 afterwards)
    type Case = (&'static str, u8, u8, [u32; 4], [u32; 4]);
    let cases: &[Case] = &[
        (
            "32_32_32_32 float xyzw",
            7,
            77,
            [f(1.5), f(-0.0), 0x7FC0_1234, 0x0000_0001],
            [f(1.5), f(-0.0), 0x7FC0_1234, 0x0000_0001],
        ),
        (
            "32_32_32_32 sint xyzw",
            7,
            76,
            [u32::MAX, 0x8000_0000, 7, 0x1234_5678],
            [u32::MAX, 0x8000_0000, 7, 0x1234_5678],
        ),
        (
            "32_32_32 float xyz",
            6,
            74,
            [f(2.0), f(3.0), f(4.0), 0xDEAD_BEEF],
            [f(2.0), f(3.0), f(4.0), 0],
        ),
        (
            "32_32 uint xy",
            5,
            63,
            [0xAABB_CCDD, 9, 0xDEAD_BEEF, 0xDEAD_BEEF],
            [0xAABB_CCDD, 9, 0, 0],
        ),
        (
            "32 float x",
            4,
            22,
            [f(-7.25), 1, 2, 3],
            [f(-7.25), 0, 0, 0],
        ),
        (
            // d16: the halves are stored as they are.
            "16_16 float d16_xy",
            13,
            29,
            [0xBC00_3555, 0xDEAD_BEEF, 0, 0],
            [0xBC00_3555, 0, 0, 0],
        ),
        (
            "16_16_16_16 float d16_xyzw",
            15,
            71,
            [0x1234_5678, 0x7C00_FC00, 0, 0],
            [0x1234_5678, 0x7C00_FC00, 0, 0],
        ),
    ];
    for &(what, op, fmt, inputs, want) in cases {
        let mut body = vec![vop1(0x001, 5, 0x80)]; // v_mov_b32 v5, 0
        body.extend(mtbuf_store(op, fmt, 1));
        let shader = build_valu_cs(&body);
        for (backend, out) in run_dwords(&shader, inputs, 8) {
            assert_eq!(
                out[4..8],
                want,
                "{backend}: tbuffer_store {what}: wrote {:#010x?}, want {want:#010x?}",
                &out[4..8],
            );
        }
    }
}

#[test]
fn tbuffer_store_format_refuses_what_it_does_not_convert() {
    for (what, op, fmt, reason) in [
        ("8_8_8_8 unorm", 7, 56, "a converted or narrow format"),
        ("16_16 uint", 5, 27, "a converted or narrow format"),
        ("16 float", 4, 13, "an element under a dword"),
        (
            "32_32_32_32 float, xy only",
            5,
            77,
            "some of its format's channels",
        ),
        ("32 float d16_x", 12, 22, "a converted or narrow format"),
        // f32 to f16 waits for nearest-even narrowing (#422).
        ("16_16 float xy", 5, 29, "a converted or narrow format"),
        (
            "16_16_16_16 float xyzw",
            7,
            71,
            "a converted or narrow format",
        ),
    ] {
        let mut body = vec![vop1(0x001, 5, 0x80)];
        body.extend(mtbuf_store(op, fmt, 1));
        match try_build_valu_cs(&body) {
            Ok(_) => panic!("tbuffer_store {what} lowered; it should be refused"),
            Err(e) => assert!(e.contains(reason), "tbuffer_store {what}: refused with {e}"),
        }
    }
}

/// Indexed (structured) loads, stores and atomics through a root UAV whose
/// title declares a stride reach byte `index * stride + offset` of the
/// buffer the root UAV binds: the host view is raw, with no stride of its own.
#[test]
fn structured_access_through_a_root_uav_reaches_index_times_stride() {
    const STRIDE: u32 = 8;
    // MUBUF `op vdata, vaddr(index), s[srsrc:srsrc+3], 0 idxen offset:off`.
    let mubuf_idxen = |op: u32, vdata: u8, vaddr: u8, srsrc: u8, off: u16| -> [u32; 2] {
        [
            0x38 << 26 | op << 18 | 1 << 13 | (off as u32 & 0xFFF),
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    };
    const STORE: u32 = 0x1C; // buffer_store_dword
    const LOAD: u32 = 0x0C; // buffer_load_dword
    const ATOMIC_ADD: u32 = 0x32; // buffer_atomic_add (no return)

    let mut words = vec![
        // A buffer descriptor around the root UAV s[0:1] in s[8:11].
        s_mov_b32(8, 0),
        s_mov_b32(9, 1),
        s_mov_b32(10, 0xC1), // -1 records
        s_mov_b32(11, 0xFF),
        0x3101_4FAC,
        vop1(0x001, 1, 128 + 2), // v1 = 2 (index)
        vop1(0x001, 2, 0xFF),    // v2 = 0xABCD
        0x0000_ABCD,
        vop1(0x001, 3, 128 + 5), // v3 = 5
        vop1(0x001, 5, 128),     // v5 = 0 (index)
    ];
    // Element 2: dword 5 = 0xABCD, dword 4 += 5; then element 0, dword 0 =
    // what element 2 byte 4 holds.
    words.extend(mubuf_idxen(STORE, 2, 1, 8, 4));
    words.extend(mubuf_idxen(ATOMIC_ADD, 3, 1, 8, 0));
    words.extend(mubuf_idxen(LOAD, 4, 1, 8, 4));
    words.extend(mubuf_idxen(STORE, 4, 5, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: STRIDE,
            dwords: 2,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let container = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&container).expect("container");
    let shader = super::emit(&c, &native, &p, &st, &io, &res).expect("emit");
    assert_eq!(shader.root_uavs, vec![0]);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(vec![0; 4])), (1, Root::Uav(8))],
                    views: &[],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(
            &out[0][..8],
            &[0xABCD, 0, 0, 0, 5, 0xABCD, 0, 0],
            "{label}: element 2 is bytes 16..24 of the buffer"
        );
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
