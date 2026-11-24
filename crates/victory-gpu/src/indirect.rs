//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is checked on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget of 2^28 work units (one per vertex or index
//! instance, 64 per compute group, 128 per mesh group). Records that do not
//! fit run empty.
//!
//! Draws and one-dimensional mesh dispatches retain their existing clamps.
//! Compute dispatches above 65,535 groups on an axis or 2^20 groups total
//! run empty. Dimensions within those limits are preserved.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
RWByteAddressBuffer args : register(u0);
// Work reserved by earlier records in the list.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 28;

bool reserve(uint cost) {
    uint before;
    budget.InterlockedAdd(0, 0, before);
    [allow_uav_condition]
    while (true) {
        if (before > BUDGET || cost > BUDGET - before) return false;
        uint observed;
        budget.InterlockedCompareExchange(0, before, before + cost, observed);
        if (observed == before) return true;
        before = observed;
    }
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    uint at = id.x * stride;
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = args.Load3(at);
        if (any(g > 65535u)) {
            // Even an empty dispatch must have legal dimensions.
            args.Store3(at, uint3(0, 0, 0));
            return;
        }
        if (any(g == 0u)) return;
        // Two legal dimensions fit in a uint. Check the third before
        // multiplying so the cost always covers the dimensions we keep.
        uint xy = g.x * g.y;
        if (xy > (1u << 20) / g.z) {
            args.Store(at, 0);
            return;
        }
        cost = xy * g.z * 64;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    if (cost != 0) {
        if (!reserve(cost)) {
            // Zero the vertex/index count or the X group count.
            args.Store(at, 0);
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

const THREADS_PER_GROUP: u32 = 64;

/// Full record span, limited by the validator's dispatch size and 32-bit
/// byte offsets. Native signatures also require a whole argument per record.
pub(crate) fn byte_len(kind: u32, stride: u32, records: u32) -> Option<u64> {
    let minimum = match kind {
        0 => 16,
        1 => 20,
        2 | 10 => 12,
        _ => return None,
    };
    if stride < minimum
        || !stride.is_multiple_of(4)
        || records.div_ceil(THREADS_PER_GROUP) > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION
    {
        return None;
    }
    let stride = u64::from(stride);
    let last = u64::from(records.saturating_sub(1)) * stride;
    if last + u64::from(minimum) > (1 << 32) {
        return None;
    }
    Some(stride * u64::from(records))
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffers ready to be read as arguments. `count`
/// is the count buffer if it is distinct from `buffer`. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    count: Option<&ID3D12Resource>,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    if !gpu.is_multiple_of(4) {
        return false;
    }
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget.GetGPUVirtualAddress());
        l.Dispatch(records.div_ceil(THREADS_PER_GROUP), 1, 1);
        l.ResourceBarrier(&[uav_barrier()]);
        for buffer in [Some(buffer), count].into_iter().flatten() {
            l.ResourceBarrier(&[transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
    }
    true
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource, count: Option<&ID3D12Resource>) {
    for buffer in [Some(buffer), count].into_iter().flatten() {
        // SAFETY: an open list and a live buffer.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                buffer,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Host;

    const COMPUTE: u32 = 2;
    const BUDGET: u32 = 1 << 28;
    const FULL_DISPATCH: [u32; 3] = [1024, 1024, 1];
    const EMPTY_DISPATCH: [u32; 3] = [0, 1024, 1];

    #[test]
    fn record_span_bounds() {
        let max = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION * THREADS_PER_GROUP;
        assert_eq!(byte_len(2, 12, max), Some(u64::from(12 * max)));
        assert_eq!(byte_len(2, 12, max + 1), None);
        assert_eq!(byte_len(2, 12, 0), Some(0));
        assert_eq!(byte_len(2, 1 << 30, 4), Some(1 << 32));
        assert_eq!(byte_len(2, 1 << 30, 5), None);
        assert_eq!(byte_len(2, 12, u32::MAX), None);
        assert_eq!(byte_len(0, 12, 1), None);
        assert_eq!(byte_len(1, 16, 1), None);
        assert_eq!(byte_len(10, 0, 1), None);
        assert_eq!(byte_len(2, 13, 1), None);
        assert_eq!(byte_len(3, 12, 1), None);
        assert_eq!(byte_len(1, 24, 2), Some(48));
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        use windows::Win32::Graphics::Dxgi::*;

        let Some(host) = host() else {
            eprintln!("skipped GPU validation: no host D3D12 device");
            return;
        };
        // SAFETY: plain queries on the live device and DXGI factory.
        let desc = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter: IDXGIAdapter1 = factory
                .EnumAdapterByLuid(host.device.GetAdapterLuid())
                .unwrap();
            adapter.GetDesc1().unwrap()
        };
        let end = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        eprintln!(
            "GPU validation adapter: {}",
            String::from_utf16_lossy(&desc.Description[..end])
        );
        assert!(clamp().is_some());
    }

    struct Arguments<const N: usize> {
        buffer: ID3D12Resource,
        mapped: *mut u32,
        records: usize,
    }

    impl<const N: usize> Arguments<N> {
        fn new(host: &Host, input: &[[u32; N]]) -> Self {
            let (buffer, mapped) = host.shared_buffer(size_of_val(input) as u64 + 4).unwrap();
            let mapped = mapped.cast::<u32>();
            // SAFETY: the mapping holds a guard word followed by all records.
            unsafe {
                mapped.write(u32::MAX);
                mapped
                    .add(1)
                    .cast::<[u32; N]>()
                    .copy_from_nonoverlapping(input.as_ptr(), input.len());
            }
            Self {
                buffer,
                mapped,
                records: input.len(),
            }
        }

        fn gpu(&self) -> u64 {
            // SAFETY: the resource is live and starts with a guard word.
            unsafe { self.buffer.GetGPUVirtualAddress() + 4 }
        }

        fn record(host: &Host, list: &HostList, kind: u32, input: &[[u32; N]]) -> Self {
            let args = Self::new(host, input);
            assert!(bound(
                list,
                &args.buffer,
                args.gpu(),
                None,
                (kind, (4 * N) as u32, input.len() as u32)
            ));
            done(list, &args.buffer, None);
            args
        }

        fn read(&self) -> Vec<[u32; N]> {
            // SAFETY: the buffer is still live and the submission has finished.
            unsafe {
                assert_eq!(self.mapped.read(), u32::MAX);
                std::slice::from_raw_parts(self.mapped.add(1).cast(), self.records).to_vec()
            }
        }
    }

    fn used(list: &HostList) -> u32 {
        let buffer = list.budget().unwrap();
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the submission has finished and the buffer holds a u32.
        unsafe {
            buffer
                .Map(
                    0,
                    Some(&D3D12_RANGE { Begin: 0, End: 4 }),
                    Some(&mut mapped),
                )
                .unwrap();
            let used = mapped.cast::<u32>().read();
            buffer.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            used
        }
    }

    #[test]
    fn compute_dimensions_and_resubmission() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let args = Arguments::record(
            host,
            &list,
            COMPUTE,
            &[
                [2, 3, 4],
                [17, 0, 3],
                FULL_DISPATCH,
                [1025, 1024, 1],
                [2048, 2048, 1024],
                [65535, 65535, 65535],
                [65536, 1, 1],
                [0, 65536, 1],
            ],
        );
        list.close();
        for _ in 0..2 {
            host.execute(&[&list]);
            let actual = args.read();
            assert_eq!(actual[4], [0, 2048, 1024], "2^32 groups must run empty");
            assert_eq!(
                actual,
                [
                    [2, 3, 4],
                    [17, 0, 3],
                    FULL_DISPATCH,
                    EMPTY_DISPATCH,
                    [0, 2048, 1024],
                    [0, 65535, 65535],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            );
            assert_eq!(used(&list), (24 + (1 << 20)) * 64);
        }
    }

    #[test]
    fn compute_work_above_the_record_limit_runs_empty() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let args = Arguments::record(host, &list, COMPUTE, &[[1025, 1024, 1]]);
        list.close();
        host.execute(&[&list]);
        assert_eq!(args.read(), [EMPTY_DISPATCH]);
        assert_eq!(used(&list), 0);
    }

    #[test]
    fn rejected_reservations_cannot_wrap_the_budget() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let full = Arguments::record(host, &list, COMPUTE, &[FULL_DISPATCH; 4]);
        let rejected = Arguments::record(host, &list, COMPUTE, &[FULL_DISPATCH; 128]);
        let last = Arguments::record(host, &list, COMPUTE, &[[1, 1, 1]]);
        list.close();
        host.execute(&[&list]);
        assert_eq!(full.read(), [FULL_DISPATCH; 4]);
        let admitted = rejected.read().iter().filter(|g| g[0] != 0).count();
        assert_eq!(admitted, 0);
        assert_eq!(last.read(), [[0, 1, 1]]);
        assert_eq!(used(&list), BUDGET);
    }

    #[test]
    fn rejected_reservation_leaves_room_for_smaller_dispatch() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let _full = Arguments::record(host, &list, COMPUTE, &[FULL_DISPATCH; 3]);
        let _almost = Arguments::record(host, &list, COMPUTE, &[[32767, 32, 1]]);
        let rejected = Arguments::record(host, &list, COMPUTE, &[[33, 1, 1]]);
        let fits = Arguments::record(host, &list, COMPUTE, &[[32, 1, 1]]);
        list.close();
        host.execute(&[&list]);
        assert_eq!(rejected.read(), [[0, 1, 1]]);
        assert_eq!(fits.read(), [[32, 1, 1]]);
        assert_eq!(used(&list), BUDGET);
    }

    #[test]
    fn concurrent_reservations_stay_within_budget() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let args = Arguments::record(host, &list, COMPUTE, &[[128, 128, 1]; 1024]);
        list.close();
        host.execute(&[&list]);
        let mut admitted = 0;
        for record in args.read() {
            match record {
                [128, 128, 1] => admitted += 1,
                [0, 128, 1] => {}
                _ => panic!("unexpected dispatch {record:?}"),
            }
        }
        assert_eq!(admitted * 128 * 128 * 64, BUDGET);
        assert_eq!(used(&list), BUDGET);
    }

    #[test]
    fn draw_and_mesh_limits_are_preserved() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let draw = Arguments::record(host, &list, 0, &[[3, 2, 5, 7], [u32::MAX, u32::MAX, 9, 11]]);
        let indexed = Arguments::record(host, &list, 1, &[[6, 2, 1, u32::MAX, 2]]);
        let mesh = Arguments::record(host, &list, 10, &[[17, 1, 1], [7, 0, 1], [65536, 3, 2]]);
        list.close();
        host.execute(&[&list]);
        assert_eq!(draw.read(), [[3, 2, 5, 7], [1 << 24, 4, 9, 11]]);
        assert_eq!(indexed.read(), [[6, 2, 1, u32::MAX, 2]]);
        assert_eq!(mesh.read(), [[17, 1, 1], [7, 0, 1], [65535, 1, 1]]);
        assert_eq!(used(&list), 6 + (1 << 26) + 12 + (17 + 65535) * 128);
    }

    #[test]
    fn validated_arguments_run_with_a_separate_count_buffer() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let args = Arguments::new(host, &[[1, 1, 1]; 2]);
        let count = Arguments::new(host, &[[1]]);
        let work = Arguments::new(host, &[[6, 2, 3]]);
        let mut signature: Option<ID3D12CommandSignature> = None;
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        // SAFETY: the signature describes one dispatch with no root arguments.
        unsafe {
            host.device
                .CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 12,
                        NumArgumentDescs: 1,
                        pArgumentDescs: &argument,
                        NodeMask: 0,
                    },
                    None::<&ID3D12RootSignature>,
                    &mut signature,
                )
                .unwrap();
        }
        let signature = signature.unwrap();
        assert!(bound(
            &list,
            &args.buffer,
            args.gpu(),
            Some(&count.buffer),
            (COMPUTE, 12, 2)
        ));
        // Use the bound pipeline as the workload and redirect it to its own
        // buffer. Its budget contribution records how many dispatches ran.
        let constants = [COMPUTE, 12, 1, 0];
        // SAFETY: all resources and the signature are live. Both input
        // buffers are in indirect-argument state, with records at byte 4.
        unsafe {
            list.list
                .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            list.list.SetComputeRootUnorderedAccessView(1, work.gpu());
            list.list
                .ExecuteIndirect(&signature, 2, &args.buffer, 4, &count.buffer, 4);
        }
        done(&list, &args.buffer, Some(&count.buffer));
        list.close();
        host.execute(&[&list]);
        assert_eq!(used(&list), (2 + 36) * 64);
    }
}
