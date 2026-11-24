//! DualSense (PlayStation 5) pads over HID.
//!
//! The console reports gamepads through GameInput; on a PC a DualSense is not
//! an XInput device, so the title never sees it. This reads the pad's HID
//! input reports and exposes the same state shape the XInput source does.
//! Both transports are parsed: USB report 0x01 and Bluetooth report 0x31.
//!
//! shortcut: touchpad, gyro and the lightbar are not reported (the title's
//! pad surface has no axes for them); rumble is not sent back yet.

#![allow(non_snake_case)]

use std::sync::{Mutex, Once};
use std::time::Duration;

use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    DIGCF_DEVICEINTERFACE, DIGCF_PRESENT, HDEVINFO, SetupDiDestroyDeviceInfoList,
    SetupDiEnumDeviceInterfaces, SetupDiGetClassDevsW, SetupDiGetDeviceInterfaceDetailW,
    SP_DEVICE_INTERFACE_DATA, SP_DEVICE_INTERFACE_DETAIL_DATA_W,
};
use windows_sys::Win32::Devices::HumanInterfaceDevice::{
    GUID_DEVINTERFACE_HID, HIDD_ATTRIBUTES, HIDP_CAPS, HIDP_STATUS_SUCCESS, HidD_FreePreparsedData,
    HidD_GetAttributes, HidD_GetPreparsedData, HidP_GetCaps,
};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile,
};

/// Gamepad buttons, the bits the title maps (same values as `gameinput`).
const MENU: u32 = 0x1;
const VIEW: u32 = 0x2;
const A: u32 = 0x4;
const B: u32 = 0x8;
const X: u32 = 0x10;
const Y: u32 = 0x20;
const UP: u32 = 0x40;
const DOWN: u32 = 0x80;
const LEFT: u32 = 0x100;
const RIGHT: u32 = 0x200;
const LEFT_SHOULDER: u32 = 0x400;
const RIGHT_SHOULDER: u32 = 0x800;
const LEFT_STICK: u32 = 0x1000;
const RIGHT_STICK: u32 = 0x2000;

/// Sony's vendor id and the DualSense / DualSense Edge product ids.
const SONY: u16 = 0x054c;
const DUALSENSE: u16 = 0x0ce6;
const DUALSENSE_EDGE: u16 = 0x0df2;

const GENERIC_READ: u32 = 0x8000_0000;

/// A pad reading: triggers 0 to 1, sticks -1 to 1 (up positive).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reading {
    pub buttons: u32,
    pub left_trigger: f32,
    pub right_trigger: f32,
    pub left_x: f32,
    pub left_y: f32,
    pub right_x: f32,
    pub right_y: f32,
}

static READING: Mutex<Option<Reading>> = Mutex::new(None);
static START: Once = Once::new();

/// The pad's newest state, starting the reader on first use.
pub fn latest() -> Option<Reading> {
    START.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("dualsense".into())
            .spawn(reader);
    });
    *READING.lock().unwrap_or_else(|e| e.into_inner())
}

fn reader() {
    loop {
        if let Some((handle, input_len)) = open_pad() {
            read_reports(handle, input_len);
            // SAFETY: our own handle.
            unsafe { CloseHandle(handle) };
        }
        *READING.lock().unwrap_or_else(|e| e.into_inner()) = None;
        // A pad that is off or out of range is looked for again.
        std::thread::sleep(Duration::from_secs(2));
    }
}

/// Reads until the pad goes away. A report that does not parse is ignored.
fn read_reports(handle: HANDLE, input_len: usize) {
    let mut buf = vec![0u8; input_len.max(78)];
    loop {
        let mut read = 0u32;
        // SAFETY: a live handle and a buffer of its report length.
        let ok = unsafe {
            ReadFile(
                handle,
                buf.as_mut_ptr().cast(),
                buf.len() as u32,
                &mut read,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 || read == 0 {
            return;
        }
        if let Some(reading) = parse(&buf[..read as usize]) {
            *READING.lock().unwrap_or_else(|e| e.into_inner()) = Some(reading);
        }
    }
}

/// Opens the first DualSense and returns it with its input report length.
fn open_pad() -> Option<(HANDLE, usize)> {
    for path in device_paths() {
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            continue;
        }
        let mut attributes: HIDD_ATTRIBUTES = unsafe { std::mem::zeroed() };
        attributes.Size = std::mem::size_of::<HIDD_ATTRIBUTES>() as u32;
        // SAFETY: a live HID handle and room for the attributes.
        let known = unsafe { HidD_GetAttributes(handle, &mut attributes) };
        if known
            && attributes.VendorID == SONY
            && (attributes.ProductID == DUALSENSE || attributes.ProductID == DUALSENSE_EDGE)
        {
            tracing::info!(
                product = format_args!("{:#06x}", attributes.ProductID),
                "DualSense connected over HID"
            );
            let input_len = report_length(handle);
            return Some((handle, input_len));
        }
        unsafe { CloseHandle(handle) };
    }
    None
}

/// The input report length the HID stack reports, or the DualSense's 78.
fn report_length(handle: HANDLE) -> usize {
    let mut preparsed: isize = 0;
    // SAFETY: a live HID handle and room for the pointer.
    if !unsafe { HidD_GetPreparsedData(handle, &mut preparsed) } {
        return 78;
    }
    let mut caps: HIDP_CAPS = unsafe { std::mem::zeroed() };
    // SAFETY: the preparations are ours until freed.
    let status = unsafe { HidP_GetCaps(preparsed, &mut caps) };
    unsafe { HidD_FreePreparsedData(preparsed) };
    if status == HIDP_STATUS_SUCCESS {
        usize::from(caps.InputReportByteLength).max(1)
    } else {
        78
    }
}

/// Every present HID interface's device path.
fn device_paths() -> Vec<Vec<u16>> {
    let mut paths = Vec::new();
    // SAFETY: the HID interface class, no enumerator or parent window.
    let set: HDEVINFO =
        unsafe { SetupDiGetClassDevsW(&GUID_DEVINTERFACE_HID, std::ptr::null(), std::ptr::null_mut(), DIGCF_PRESENT | DIGCF_DEVICEINTERFACE) };
    if set == -1 {
        return paths;
    }
    let mut index = 0u32;
    loop {
        let mut interface: SP_DEVICE_INTERFACE_DATA = unsafe { std::mem::zeroed() };
        interface.cbSize = std::mem::size_of::<SP_DEVICE_INTERFACE_DATA>() as u32;
        // SAFETY: the set from above, and room for one interface.
        if unsafe {
            SetupDiEnumDeviceInterfaces(
                set,
                std::ptr::null(),
                &GUID_DEVINTERFACE_HID,
                index,
                &mut interface,
            )
        } == 0
        {
            break;
        }
        index += 1;
        let mut required = 0u32;
        // SAFETY: asks for the room the detail needs.
        unsafe {
            SetupDiGetDeviceInterfaceDetailW(
                set,
                &interface,
                std::ptr::null_mut(),
                0,
                &mut required,
                std::ptr::null_mut(),
            )
        };
        if required == 0 {
            continue;
        }
        let mut buffer = vec![0u8; required as usize];
        let detail = buffer.as_mut_ptr().cast::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>();
        // SAFETY: the buffer is the size the API asked for; cbSize is its header.
        unsafe {
            (*detail).cbSize = std::mem::size_of::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>() as u32;
            if SetupDiGetDeviceInterfaceDetailW(
                set,
                &interface,
                detail,
                required,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ) == 0
            {
                continue;
            }
        }
        // The path starts after the u32 cbSize, as UTF-16.
        let path = unsafe { buffer.as_ptr().add(4).cast::<u16>() };
        let mut len = 0usize;
        // SAFETY: the API wrote a NUL-terminated path into the buffer.
        while unsafe { *path.add(len) } != 0 {
            len += 1;
        }
        paths.push(unsafe { std::slice::from_raw_parts(path, len) }.to_vec());
    }
    // SAFETY: the set is ours.
    unsafe { SetupDiDestroyDeviceInfoList(set) };
    paths
}

/// The pad state in a DualSense input report, if this one is one.
fn parse(report: &[u8]) -> Option<Reading> {
    // USB report 0x01 carries the state from byte 1; Bluetooth report 0x31 has
    // a sequence byte at 1 and the same state from byte 2.
    let base = match report.first()? {
        0x01 => 0,
        0x31 => 1,
        _ => return None,
    };
    if report.len() < base + 10 {
        return None;
    }
    let axis = |i: usize| f32::from(report[base + i]);
    let centered = |v: f32| ((v - 127.5) / 127.5).clamp(-1.0, 1.0);
    let buttons0 = report[base + 7];
    let buttons1 = report[base + 8];
    let buttons2 = report[base + 9];
    let mut buttons = 0u32;
    // The low nibble is the hat: 0 up, clockwise, 8 none.
    buttons |= match buttons0 & 0x0f {
        0 => UP,
        1 => UP | RIGHT,
        2 => RIGHT,
        3 => DOWN | RIGHT,
        4 => DOWN,
        5 => DOWN | LEFT,
        6 => LEFT,
        7 => UP | LEFT,
        _ => 0,
    };
    // The high nibble: square, cross, circle, triangle.
    if buttons0 & 0x10 != 0 {
        buttons |= X;
    }
    if buttons0 & 0x20 != 0 {
        buttons |= A;
    }
    if buttons0 & 0x40 != 0 {
        buttons |= B;
    }
    if buttons0 & 0x80 != 0 {
        buttons |= Y;
    }
    if buttons1 & 0x01 != 0 {
        buttons |= LEFT_SHOULDER;
    }
    if buttons1 & 0x02 != 0 {
        buttons |= RIGHT_SHOULDER;
    }
    if buttons1 & 0x10 != 0 {
        buttons |= VIEW; // create
    }
    if buttons1 & 0x20 != 0 {
        buttons |= MENU; // options
    }
    if buttons1 & 0x40 != 0 {
        buttons |= LEFT_STICK;
    }
    if buttons1 & 0x80 != 0 {
        buttons |= RIGHT_STICK;
    }
    if buttons2 & 0x01 != 0 {
        buttons |= MENU; // PS
    }
    Some(Reading {
        buttons,
        left_trigger: axis(5) / 255.0,
        right_trigger: axis(6) / 255.0,
        left_x: centered(axis(1)),
        // The pad's Y axis grows downwards; the title's sticks grow upwards.
        left_y: -centered(axis(2)),
        right_x: centered(axis(3)),
        right_y: -centered(axis(4)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usb_report_maps_buttons_and_axes() {
        let mut report = [0u8; 64];
        report[0] = 0x01;
        report[1] = 255; // left stick right
        report[2] = 0; // left stick up
        report[3] = 128;
        report[4] = 128;
        report[5] = 255; // left trigger
        report[7] = 0x20 | 0x2; // cross, hat right
        report[8] = 0x01 | 0x20; // L1, options
        let r = parse(&report).expect("a USB report");
        assert_eq!(r.buttons, A | RIGHT | LEFT_SHOULDER | MENU);
        assert_eq!(r.left_trigger, 1.0);
        assert_eq!(r.left_x, 1.0);
        assert_eq!(r.left_y, 1.0);
    }

    #[test]
    fn bluetooth_report_has_one_extra_byte() {
        let mut report = [0u8; 78];
        report[0] = 0x31;
        report[1] = 7; // sequence
        report[2] = 0; // left stick left
        report[3] = 255; // left stick down
        report[7] = 0xff; // right trigger
        report[8] = 0x08; // cross? no: bit 3 is square? high nibble is byte & 0xf0
        report[9] = 0;
        let r = parse(&report).expect("a Bluetooth report");
        assert_eq!(r.left_x, -1.0);
        assert_eq!(r.left_y, -1.0);
        assert_eq!(r.right_trigger, 1.0);
    }

    #[test]
    fn a_short_or_foreign_report_is_ignored() {
        assert!(parse(&[0x01, 0, 0]).is_none());
        assert!(parse(&[0x02; 64]).is_none());
    }
}
