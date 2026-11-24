//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Option<(usize, Vec<u8>)>>,
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

#[cfg(windows)]
impl Drop for Shadow {
    fn drop(&mut self) {
        crate::host::release_late(self.buffer.clone());
    }
}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    #[cfg(test)]
    pub(crate) fn read_u32(&self, offset: usize) -> u32 {
        assert!(offset + 4 <= self.len as usize);
        // SAFETY: `offset..offset + 4` lies within the mapped host buffer.
        unsafe { self.data.add(offset).cast::<u32>().read_unaligned() }
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// The title memory this host copy stands for: start address, bytes.
    pub(crate) fn bounds(&self) -> (usize, usize) {
        (self.start as usize, self.len as usize)
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &*before
            && (*at, at + bytes.len()) == (start, end)
        {
            return;
        }
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        match before.take() {
            Some((at, bytes)) => {
                grown.extend_from_slice(host(start, at));
                grown.extend_from_slice(&bytes);
                grown.extend_from_slice(host(at + bytes.len(), end));
                give_back(bytes);
            }
            None => grown.extend_from_slice(host(start, end)),
        }
        *before = Some((start, grown));
    }

    /// The runs `(first, last + 1)` of copy-back pages to ask about: the
    /// pages `synced` inside `[first, last)`, coalesced. Pages the GPU did
    /// not write keep their write-watch bits for the next refresh.
    fn copy_back_runs(synced: &[usize], first: usize, last: usize) -> Vec<(usize, usize)> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for &page in synced {
            if !(first..last).contains(&page) {
                continue;
            }
            match runs.last_mut() {
                Some((_, end)) if *end == page => *end += 1,
                Some((_, end)) if *end < page => runs.push((page, page + 1)),
                Some(_) => {}
                None => runs.push((page, page + 1)),
            }
        }
        runs
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let title = &mut title[..n];
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let mut changed = false;
                    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                        if *h != *o && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    if changed {
                        crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            // Ask only over the pages this copy back actually synced: a
            // declared view can span hundreds of MB while the GPU changed a
            // few 64 KiB pieces, and asking the kernel about the whole view
            // scanned every page of it on every submission.
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            {
                synced.sort_unstable();
                synced.dedup();
                let mut asks = Vec::new();
                for (from_page, to_page) in Self::copy_back_runs(&synced, first, last) {
                    if let Some((now, _)) =
                        crate::watch::writes(from_page * PAGE, to_page * PAGE, true)
                    {
                        asks.push((from_page, to_page, now));
                    }
                }
                if !asks.is_empty() {
                    let base = start / PAGE;
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    for (from_page, to_page, now) in asks {
                        for page in from_page..to_page {
                            copied[page - base] = now;
                        }
                    }
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty()
            && let Some((_, bytes)) = snapshot.take()
        {
            give_back(bytes);
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// The runs of pages of `[from, to)` whose host copy is missing (`0`) or
    /// older than the last write `written` reports for them, and the pages
    /// wholly inside the range, which a copy makes current. `base` is the
    /// page number of `bounds`'s start, the copy's title memory.
    fn stale_runs(
        copied: &[u64],
        base: usize,
        from: usize,
        to: usize,
        written: &[(usize, u64)],
        bounds: (usize, usize),
    ) -> (Vec<(usize, usize)>, Vec<usize>) {
        let (start, end) = bounds;
        let mut writes = written.iter().peekable();
        let (mut runs, mut fresh) = (Vec::new(), Vec::new());
        let mut run: Option<usize> = None;
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            while matches!(writes.peek(), Some(&&(p, _)) if p < page) {
                writes.next();
            }
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            if copied[page - base] == 0 || write > copied[page - base] {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    fresh.push(page);
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                runs.push((first, page));
            }
        }
        if let Some(first) = run {
            runs.push((first, last + 1));
        }
        (runs, fresh)
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let bounds = (self.start as usize, (self.start + self.len) as usize);
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let (runs, fresh) = Shadow::stale_runs(&copied, base, from, to, written, bounds);
        for page in fresh {
            copied[page - base] = now;
        }
        let mut bytes = 0;
        for (first, last) in runs {
            // Cut to the range: next to it shaders may be writing the copy.
            bytes += self.copy((first * PAGE).max(from), (last * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    if address < 0x10000 {
        return None;
    }
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let end = end.max(address.saturating_add(64 << 10));
    let len = end.checked_next_multiple_of(WINDOW)? - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    if !crate::texture::committed(address as *const u8, 1) {
        return None;
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A title write between two submissions: the second asking's journal
    /// entry is newer than the copy's watermark, so only that page copies.
    #[test]
    fn a_write_between_two_submissions_copies_only_that_page() {
        let base = 100;
        let (start, end) = (base * PAGE, (base + 4) * PAGE);
        let bounds = (start, end);
        let mut copied = [0u64; 4];
        let (runs, fresh) = Shadow::stale_runs(&copied, base, start, end, &[], bounds);
        assert_eq!(runs, [(100, 104)]);
        assert_eq!(fresh, [100, 101, 102, 103]);
        for page in fresh {
            copied[page - base] = 5; // copied at asking 5
        }

        // Nothing written: the next submission copies nothing.
        let (runs, fresh) = Shadow::stale_runs(&copied, base, start, end, &[], bounds);
        assert!(runs.is_empty() && fresh.is_empty());

        // A title thread writes page 102 between submissions; the next
        // asking reports it at 6, and only that page copies.
        let (runs, fresh) = Shadow::stale_runs(&copied, base, start, end, &[(102, 6)], bounds);
        assert_eq!(runs, [(102, 103)]);
        assert_eq!(fresh, [102]);
    }

    /// Two host copies over the same title memory (overlapping shadow
    /// buffers) keep separate watermarks. Both see the journal's write; one
    /// that copied after the write skips it, one that copied the range
    /// before the write copies just that page, and one that never copied
    /// takes the whole range.
    #[test]
    fn overlapping_copies_see_the_same_journal_write() {
        let base = 200;
        let (start, end) = (base * PAGE, (base + 8) * PAGE);
        let bounds = (start, end);
        let mut early = [0u64; 8];
        let mut before = [0u64; 8];
        let never = [0u64; 8];
        let write = [(204, 7)];
        let (_, fresh) = Shadow::stale_runs(&early, base, start, end, &[], bounds);
        for page in fresh {
            early[page - base] = 7; // copied after the write's asking
        }
        let (_, fresh) = Shadow::stale_runs(&before, base, start, end, &[], bounds);
        for page in fresh {
            before[page - base] = 6; // copied before the write's asking
        }

        let range = (start + 2 * PAGE, start + 6 * PAGE);
        let (runs, fresh) = Shadow::stale_runs(&early, base, range.0, range.1, &write, bounds);
        assert!(runs.is_empty() && fresh.is_empty(), "already current");

        let (runs, fresh) = Shadow::stale_runs(&before, base, range.0, range.1, &write, bounds);
        assert_eq!(runs, [(204, 205)]);
        assert_eq!(fresh, [204]);

        let (runs, fresh) = Shadow::stale_runs(&never, base, range.0, range.1, &write, bounds);
        assert_eq!(runs, [(202, 206)]);
        assert_eq!(fresh, [202, 203, 204, 205]);
    }

    #[test]
    fn copy_back_watches_only_the_pages_the_gpu_changed() {
        // A 100-page declared range with five changed pages: the kernel is
        // asked about those, not the other 95 pages (a 174 MB view scans
        // only the few pages the shaders wrote).
        let synced = [101, 102, 103, 150, 199];
        assert_eq!(
            Shadow::copy_back_runs(&synced, 100, 200),
            [(101, 104), (150, 151), (199, 200)]
        );
        // Nothing changed: no watch at all.
        assert!(Shadow::copy_back_runs(&[], 100, 200).is_empty());
        // Partly covered boundary pages are left out, as before.
        let synced = [99, 100, 101, 199, 200];
        assert_eq!(
            Shadow::copy_back_runs(&synced, 100, 200),
            [(100, 102), (199, 200)]
        );
        // Duplicate pages from overlapping syncs do not add runs.
        let synced = [100, 100, 101, 101];
        assert_eq!(Shadow::copy_back_runs(&synced, 100, 200), [(100, 102)]);
    }

    #[test]
    fn a_partial_page_stays_stale_outside_the_range() {
        let base = 300;
        let (start, end) = (base * PAGE, (base + 2) * PAGE);
        let bounds = (start, end);
        let copied = [0u64; 2];
        // One byte into page 300 through one byte of page 301: both pages
        // are copied in part, so neither is marked current.
        let (runs, fresh) =
            Shadow::stale_runs(&copied, base, start + 1, start + PAGE + 1, &[], bounds);
        assert_eq!(runs, [(300, 302)]);
        assert!(fresh.is_empty());
        // A whole page is marked current after its copy.
        let (_, fresh) = Shadow::stale_runs(&copied, base, start, start + PAGE, &[], bounds);
        assert_eq!(fresh, [300]);
    }
}
