// Copyright 2018-2026 the Deno authors. MIT license.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::Once;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use futures::task::AtomicWaker;

use super::bindings;
use super::snapshot;
use super::snapshot::V8Snapshot;

/// Extract the raw isolate address from an `UnsafeRawIsolatePtr`.
///
/// `UnsafeRawIsolatePtr` is `#[repr(transparent)]` over `*mut RealIsolate`,
/// so its bit-pattern is a single pointer-sized value. We use transmute
/// because the inner field is private.
///
/// The compile-time assert below guarantees the layout assumption holds.
const _: () = assert!(
  std::mem::size_of::<v8::UnsafeRawIsolatePtr>()
    == std::mem::size_of::<usize>()
);

pub(crate) fn isolate_ptr_to_key(ptr: v8::UnsafeRawIsolatePtr) -> usize {
  // SAFETY: UnsafeRawIsolatePtr is #[repr(transparent)] over *mut RealIsolate,
  // which is pointer-sized. The compile-time assert above guarantees this.
  unsafe { std::mem::transmute::<v8::UnsafeRawIsolatePtr, usize>(ptr) }
}

/// Thread-safe queue of V8 foreground tasks, shared between the global
/// isolate registry (written by V8 background threads) and the event
/// loop (drained on the main thread). Cloning is cheap (Arc).
pub type ForegroundTaskQueue = std::sync::Arc<Mutex<Vec<v8::Task>>>;

/// Per-isolate state stored in the global registry. Kept minimal: just
/// enough for platform callbacks (which only have an isolate pointer) to
/// push tasks and wake the event loop.
struct IsolateEntry {
  waker: std::sync::Arc<AtomicWaker>,
  handle: tokio::runtime::Handle,
  tasks: ForegroundTaskQueue,
}

static ISOLATE_ENTRIES: std::sync::LazyLock<
  Mutex<HashMap<usize, IsolateEntry>>,
> = std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Register an isolate in the global platform registry. The `tasks`
/// queue is shared with `JsRuntimeState` so the event loop drains it
/// directly without touching the global map.
pub fn register_isolate(
  isolate_ptr: usize,
  waker: std::sync::Arc<AtomicWaker>,
  handle: tokio::runtime::Handle,
  tasks: ForegroundTaskQueue,
) {
  let mut map = ISOLATE_ENTRIES.lock().unwrap();
  map.insert(
    isolate_ptr,
    IsolateEntry {
      waker,
      handle,
      tasks,
    },
  );
}

pub fn unregister_isolate(isolate_ptr: usize) {
  let mut map = ISOLATE_ENTRIES.lock().unwrap();
  map.remove(&isolate_ptr);
}

/// Queue an immediate foreground task and wake the event loop.
fn queue_task(key: usize, task: v8::Task) {
  let map = ISOLATE_ENTRIES.lock().unwrap();
  if let Some(entry) = map.get(&key) {
    entry.tasks.lock().unwrap().push(task);
    entry.waker.wake();
  }
}

/// Spawn a delayed V8 foreground task on the isolate's tokio runtime.
/// After the delay, the task is queued for synchronous draining (not
/// run directly on the tokio worker thread).
fn spawn_delayed_task(key: usize, task: v8::Task, delay_in_seconds: f64) {
  let map = ISOLATE_ENTRIES.lock().unwrap();
  if let Some(entry) = map.get(&key) {
    let tasks = entry.tasks.clone();
    let waker = entry.waker.clone();
    entry.handle.spawn(async move {
      tokio::time::sleep(Duration::from_secs_f64(delay_in_seconds)).await;
      tasks.lock().unwrap().push(task);
      waker.wake();
    });
  }
}

/// Custom V8 platform implementation that queues immediate foreground
/// tasks for synchronous draining, and spawns delayed tasks on tokio.
struct DenoPlatformImpl;

impl v8::PlatformImpl for DenoPlatformImpl {
  fn post_task(&self, isolate_ptr: *mut c_void, task: v8::Task) {
    queue_task(isolate_ptr as usize, task);
  }

  fn post_non_nestable_task(&self, isolate_ptr: *mut c_void, task: v8::Task) {
    queue_task(isolate_ptr as usize, task);
  }

  fn post_delayed_task(
    &self,
    isolate_ptr: *mut c_void,
    task: v8::Task,
    delay_in_seconds: f64,
  ) {
    spawn_delayed_task(isolate_ptr as usize, task, delay_in_seconds);
  }

  fn post_non_nestable_delayed_task(
    &self,
    isolate_ptr: *mut c_void,
    task: v8::Task,
    delay_in_seconds: f64,
  ) {
    spawn_delayed_task(isolate_ptr as usize, task, delay_in_seconds);
  }

  fn post_idle_task(&self, _isolate_ptr: *mut c_void, _task: v8::IdleTask) {
    unreachable!();
  }
}

fn v8_init(
  v8_platform: Option<v8::SharedRef<v8::Platform>>,
  snapshot: bool,
  expose_natives: bool,
) {
  #[cfg(feature = "include_icu_data")]
  {
    v8::icu::set_common_data_77(deno_core_icudata::ICU_DATA).unwrap();
  }

  let base_flags = concat!(
    " --wasm-test-streaming",
    " --no-validate-asm",
    " --turbo_fast_api_calls",
    " --harmony-temporal",
    " --js-float16array",
    " --js-explicit-resource-management",
    " --js-source-phase-imports",
    " --js-defer-import-eval"
  );
  let snapshot_flags = "--predictable --random-seed=42";
  let expose_natives_flags = "--expose_gc --allow_natives_syntax";
  let lazy_flags = if cfg!(feature = "snapshot_flags_eager_parse") {
    "--no-lazy --no-lazy-eval --no-lazy-streaming"
  } else {
    ""
  };
  let flags = match (snapshot, expose_natives) {
    (false, false) => base_flags.to_string(),
    (true, false) => {
      format!("{base_flags} {snapshot_flags} {lazy_flags}")
    }
    (false, true) => format!("{base_flags} {expose_natives_flags}"),
    (true, true) => {
      format!(
        "{base_flags} {snapshot_flags} {lazy_flags} {expose_natives_flags}"
      )
    }
  };
  v8::V8::set_flags_from_string(&flags);

  let v8_platform = v8_platform.unwrap_or_else(|| {
    let unprotected =
      cfg!(any(test, feature = "unsafe_use_unprotected_platform"));
    // Cap V8 platform thread pool to 4 threads (like Node.js).
    // Using all available cores (the default when 0 is passed) wastes
    // memory for workloads that rarely use background V8 tasks.
    let thread_pool_size = std::cmp::min(
      std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4),
      4,
    );
    v8::new_custom_platform(
      thread_pool_size,
      false,
      unprotected,
      DenoPlatformImpl,
    )
    .make_shared()
  });
  v8::V8::initialize_platform(v8_platform.clone());
  v8::V8::initialize();
}

pub fn init_v8(
  v8_platform: Option<v8::SharedRef<v8::Platform>>,
  snapshot: bool,
  expose_natives: bool,
) {
  static DENO_INIT: Once = Once::new();
  static DENO_SNAPSHOT: AtomicBool = AtomicBool::new(false);
  static DENO_SNAPSHOT_SET: AtomicBool = AtomicBool::new(false);

  if DENO_SNAPSHOT_SET.load(Ordering::SeqCst) {
    let current = DENO_SNAPSHOT.load(Ordering::SeqCst);
    assert_eq!(
      current, snapshot,
      "V8 may only be initialized once in either snapshotting or non-snapshotting mode. Either snapshotting or non-snapshotting mode may be used in a single process, not both."
    );
    DENO_SNAPSHOT_SET.store(true, Ordering::SeqCst);
    DENO_SNAPSHOT.store(snapshot, Ordering::SeqCst);
  }

  DENO_INIT.call_once(move || v8_init(v8_platform, snapshot, expose_natives));
}

/// MADV_DONTNEED a byte range that lives in a private, read-only file mapping
/// (e.g. data baked into `.rodata` via `include_bytes!` or referenced via a
/// `&'static [u8]`). The kernel drops the resident pages immediately;
/// subsequent reads re-fault them from the backing file.
///
/// On macOS / Windows / non-unix, this is a no-op — Linux's MADV_DONTNEED
/// semantics for file-backed mappings don't carry over.
#[cfg(all(unix, not(target_os = "macos")))]
fn drop_resident_pages_in_rodata(ptr: usize, len: usize) {
  // Page-align: madvise rejects unaligned starts, and we round the end down
  // so we never drop bytes outside the caller's range.
  let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
  if page == 0 {
    return;
  }
  let mask = page - 1;
  let start = (ptr + mask) & !mask;
  let end = (ptr + len) & !mask;
  if end <= start {
    return;
  }
  // SAFETY: caller guarantees [ptr, ptr+len) lies entirely within a
  // private read-only mapping. MADV_DONTNEED on such mappings drops
  // resident pages without touching the file; on next access the
  // kernel re-faults from disk.
  unsafe {
    libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_DONTNEED);
  }
}

/// **Experimental**: MADV_DONTNEED the entire executable code (.text) range
/// of the main binary, gated on `DENO_RSS_PROBE_DROP_TEXT=1`.
///
/// The kernel re-faults executable pages from the binary file on next fetch,
/// so this is functionally safe but will cause a short re-fault storm as
/// the runtime executes its hot paths. The interesting measurement is the
/// *steady-state* RSS that follows — the gap to the un-probed baseline is
/// code that was paged in once during init and never used again.
#[cfg(target_os = "linux")]
#[allow(
  clippy::disallowed_methods,
  reason = "experimental probe gated on env var; sys_traits is not threaded \
            into deno_core's runtime setup path"
)]
fn drop_text_pages_probe() {
  use std::ffi::c_void;
  if std::env::var_os("DENO_RSS_PROBE_DROP_TEXT").is_none() {
    return;
  }

  struct Out {
    text_start: usize,
    text_size: usize,
    found: bool,
  }
  extern "C" fn cb(
    info: *mut libc::dl_phdr_info,
    _size: libc::size_t,
    data: *mut c_void,
  ) -> libc::c_int {
    // first entry = main executable
    let info = unsafe { &*info };
    let out = unsafe { &mut *(data as *mut Out) };
    let phdrs = unsafe {
      std::slice::from_raw_parts(info.dlpi_phdr, info.dlpi_phnum as usize)
    };
    for phdr in phdrs {
      // PT_LOAD with PF_R | PF_X (= 1 | 4 = 5)
      if phdr.p_type == libc::PT_LOAD && phdr.p_flags == 5 {
        out.text_start = info.dlpi_addr as usize + phdr.p_vaddr as usize;
        out.text_size = phdr.p_memsz as usize;
        out.found = true;
        break;
      }
    }
    1
  }
  let mut out = Out {
    text_start: 0,
    text_size: 0,
    found: false,
  };
  // SAFETY: dl_iterate_phdr is safe; cb is a valid C callback.
  unsafe {
    libc::dl_iterate_phdr(Some(cb), &mut out as *mut _ as *mut c_void);
  }
  if !out.found {
    return;
  }
  let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
  if page == 0 {
    return;
  }
  let mask = page - 1;
  let start = (out.text_start + mask) & !mask;
  let end = (out.text_start + out.text_size) & !mask;
  if end <= start {
    return;
  }
  // SAFETY: range is the executable LOAD segment of the main binary, which
  // is private and file-backed. MADV_DONTNEED drops resident pages; future
  // instruction fetches re-fault from disk.
  unsafe {
    libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_DONTNEED);
  }
}

/// Drop the resident pages backing the binary's `.rela.dyn` relocation table.
///
/// Once the dynamic linker has applied relocations (by the time we run any
/// Rust code), this table is dead — but it stays mapped (and resident, since
/// it's read sequentially at startup) for the life of the process. On a
/// release-lite deno binary it costs ~8 MB of RSS for nothing.
///
/// We locate the table by walking PT_DYNAMIC for our own executable and
/// reading DT_RELA / DT_RELASZ. The pages are read-only and file-backed;
/// MADV_DONTNEED frees them immediately, and any unexpected re-access would
/// re-fault from disk.
#[cfg(target_os = "linux")]
fn drop_resident_pages_in_relocation_table() {
  use std::ffi::c_void;

  // ELF dynamic-section tags. libc doesn't expose these.
  const DT_NULL: i64 = 0;
  const DT_RELA: i64 = 7;
  const DT_RELASZ: i64 = 8;

  #[repr(C)]
  #[derive(Clone, Copy)]
  struct Elf64Dyn {
    d_tag: i64,
    d_val: u64,
  }

  struct Out {
    load_base: usize,
    rela_addr: u64,
    rela_size: u64,
    found: bool,
  }

  extern "C" fn callback(
    info: *mut libc::dl_phdr_info,
    _size: libc::size_t,
    data: *mut c_void,
  ) -> libc::c_int {
    // The first entry from dl_iterate_phdr is always the main executable
    // (dlpi_name == ""). Take it and stop iterating.
    // SAFETY: dl_iterate_phdr guarantees `info` is valid for the duration
    // of the callback, and `data` is the pointer we passed in.
    let info = unsafe { &*info };
    let out = unsafe { &mut *(data as *mut Out) };
    out.load_base = info.dlpi_addr as usize;
    // SAFETY: dlpi_phdr/dlpi_phnum describe the program header table.
    let phdrs = unsafe {
      std::slice::from_raw_parts(info.dlpi_phdr, info.dlpi_phnum as usize)
    };
    for phdr in phdrs {
      if phdr.p_type != libc::PT_DYNAMIC {
        continue;
      }
      let dyn_addr = info.dlpi_addr as usize + phdr.p_vaddr as usize;
      let dyn_count = phdr.p_memsz as usize / std::mem::size_of::<Elf64Dyn>();
      // SAFETY: PT_DYNAMIC is loaded into memory at dlpi_addr + p_vaddr;
      // p_memsz bounds the table.
      let entries = unsafe {
        std::slice::from_raw_parts(dyn_addr as *const Elf64Dyn, dyn_count)
      };
      for entry in entries {
        match entry.d_tag {
          DT_NULL => break,
          DT_RELA => out.rela_addr = entry.d_val,
          DT_RELASZ => out.rela_size = entry.d_val,
          _ => {}
        }
      }
      break;
    }
    out.found = true;
    1 // stop after the main executable
  }

  let mut out = Out {
    load_base: 0,
    rela_addr: 0,
    rela_size: 0,
    found: false,
  };
  // SAFETY: callback is a valid C function pointer; we pass a stack pointer
  // for `data` that outlives the synchronous call.
  unsafe {
    libc::dl_iterate_phdr(Some(callback), &mut out as *mut _ as *mut c_void);
  }

  if !out.found || out.rela_addr == 0 || out.rela_size == 0 {
    return;
  }

  // For a PIE binary GNU ld stores DT_RELA as a link-time VA (typically a
  // small offset like 0x6e80). glibc's ld.so applies dlpi_addr internally
  // when reading the table; the value in the dynamic section may or may not
  // already be relocated. If it's smaller than the load base, treat it as
  // a link-time VA and adjust.
  let rela_addr = if (out.rela_addr as usize) < out.load_base {
    out.load_base + out.rela_addr as usize
  } else {
    out.rela_addr as usize
  };

  drop_resident_pages_in_rodata(rela_addr, out.rela_size as usize);
}

pub fn create_isolate(
  will_snapshot: bool,
  maybe_create_params: Option<v8::CreateParams>,
  maybe_startup_snapshot: Option<V8Snapshot>,
  external_refs: Cow<'static, [v8::ExternalReference]>,
) -> v8::OwnedIsolate {
  let mut params = maybe_create_params.unwrap_or_default();
  let mut isolate = if will_snapshot {
    snapshot::create_snapshot_creator(
      external_refs,
      maybe_startup_snapshot,
      params,
    )
  } else {
    params = params.external_references(external_refs);
    let has_snapshot = maybe_startup_snapshot.is_some();
    // Capture snapshot byte range so we can release its physical pages
    // after V8 has finished deserialization. The snapshot blob is embedded
    // in the binary's .rodata section; once V8 has read it, the kernel can
    // drop those pages — they'll re-fault from the binary file if V8 ever
    // re-reads them (which it shouldn't after isolate startup).
    let snapshot_range = maybe_startup_snapshot
      .as_ref()
      .map(|s| (s.0.as_ptr() as usize, s.0.len()));
    if let Some(snapshot) = maybe_startup_snapshot {
      params = params.snapshot_blob(v8::StartupData::from(snapshot.0));
    }
    static FIRST_SNAPSHOT_INIT: AtomicBool = AtomicBool::new(false);
    static SNAPSHOW_INIT_MUT: Mutex<()> = Mutex::new(());

    // On Windows, the snapshot deserialization code appears to be crashing and we are not
    // certain of the reason. We take a mutex the first time an isolate with a snapshot to
    // prevent this. https://github.com/denoland/deno/issues/15590
    let res = if cfg!(windows)
      && has_snapshot
      && FIRST_SNAPSHOT_INIT.load(Ordering::SeqCst)
    {
      let _g = SNAPSHOW_INIT_MUT.lock().unwrap();
      let res = v8::Isolate::new(params);
      FIRST_SNAPSHOT_INIT.store(true, Ordering::SeqCst);
      res
    } else {
      v8::Isolate::new(params)
    };

    #[cfg(all(unix, not(target_os = "macos")))]
    if let Some((ptr, len)) = snapshot_range {
      drop_resident_pages_in_rodata(ptr, len);
    }

    // The dynamic linker has long since applied all relocations, so the
    // .rela.dyn table is dead but still resident — drop those pages too.
    // Done once per process; subsequent isolates re-do it as a no-op
    // (already-evicted pages stay evicted).
    #[cfg(target_os = "linux")]
    {
      use std::sync::Once;
      static DROP_RELA: Once = Once::new();
      DROP_RELA.call_once(drop_resident_pages_in_relocation_table);
      static DROP_TEXT: Once = Once::new();
      DROP_TEXT.call_once(drop_text_pages_probe);
    }

    res
  };

  isolate.set_microtasks_policy(v8::MicrotasksPolicy::Explicit);
  isolate.set_capture_stack_trace_for_uncaught_exceptions(true, 10);
  isolate.set_promise_reject_callback(bindings::promise_reject_callback);
  isolate.set_prepare_stack_trace_callback(
    crate::error::prepare_stack_trace_callback,
  );
  isolate.set_host_initialize_import_meta_object_callback(
    bindings::host_initialize_import_meta_object_callback,
  );
  isolate.set_host_import_module_dynamically_callback(
    bindings::host_import_module_dynamically_callback,
  );
  isolate.set_host_import_module_with_phase_dynamically_callback(
    bindings::host_import_module_with_phase_dynamically_callback,
  );
  isolate.set_wasm_async_resolve_promise_callback(
    bindings::wasm_async_resolve_promise_callback,
  );

  isolate
}
