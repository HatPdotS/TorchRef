//! A dedicated rayon pool, sized per call from torch's intra-op thread count.
//!
//! Dedicated rather than rayon's global pool so the size follows `torch.set_num_threads`
//! (and therefore `TORCHREF_NUM_THREADS`) instead of the machine's core count, which on a
//! shared cluster node is the wrong number.

use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::{Arc, Mutex};

static POOL: Mutex<Option<(usize, Arc<ThreadPool>)>> = Mutex::new(None);

/// A pool of `n_threads` workers, rebuilt only when the requested size changes.
/// `n_threads == 0` means "all available cores".
pub fn get(n_threads: usize) -> Arc<ThreadPool> {
    let n = if n_threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        n_threads
    };
    let mut guard = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((size, pool)) = guard.as_ref() {
        if *size == n {
            return Arc::clone(pool);
        }
    }
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("torchref-kernels-{i}"))
            .build()
            .expect("failed to start the kernel thread pool"),
    );
    *guard = Some((n, Arc::clone(&pool)));
    pool
}
