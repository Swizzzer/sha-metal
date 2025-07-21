// src/main.rs
use indicatif::{ProgressBar, ProgressStyle};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

#[repr(C)]
#[derive(Debug)]
struct GPUInfo {
    core_count: c_int,
    max_threads_per_threadgroup: c_int,
    name: [c_char; 256],
}

impl Default for GPUInfo {
    fn default() -> Self {
        Self {
            core_count: 0,
            max_threads_per_threadgroup: 0,
            name: [0; 256],
        }
    }
}

#[link(name = "metal_bridge", kind = "static")]
unsafe extern "C" {
    fn init_metal(gpu_info: *mut GPUInfo) -> c_int;
    fn search_on_gpu(
        start_index: u64,
        count: u64,
        target: *const u8,
        result: *mut u8,
        max_threads_per_threadgroup: c_int,
    ) -> c_int;
    fn cleanup_metal();
}

fn main() {
    let tar_hex = "f49cf6381e322b147053b74e4500af8533ac1e4c";
    let tar_bytes = hex::decode(tar_hex).expect("Failed to decode target hash");

    println!("Initializing Metal GPU...");
    let mut gpu_info = GPUInfo::default();
    // Unsafe block is required for FFI calls.
    unsafe {
        if init_metal(&mut gpu_info) != 0 {
            panic!("Metal initialization failed");
        }
    }

    let gpu_name = unsafe { CStr::from_ptr(gpu_info.name.as_ptr()) }.to_string_lossy();
    let gpu_cores = gpu_info.core_count as u64;
    let max_threads_per_threadgroup = gpu_info.max_threads_per_threadgroup as u64;

    let gpu_batch_size = (gpu_cores * max_threads_per_threadgroup * 64).min(1 << 24);

    println!("\n=== Rust/Metal Configuration ===");
    println!("GPU: {}", gpu_name);
    println!("GPU Cores: {}", gpu_cores);
    println!("Max Threads Per Group: {}", max_threads_per_threadgroup);
    println!(
        "Batch Size: {} ({:.2}M hashes)",
        gpu_batch_size,
        gpu_batch_size as f64 / 1_048_576.0
    );
    println!("\nMetal GPU Initialized Successfully!");

    let total_operations = 16u64.pow(8);
    let num_schedulers = num_cpus::get().min(4);

    let global_search_index = Arc::new(AtomicU64::new(0));
    let found_flag = Arc::new(AtomicBool::new(false));
    let found_result = Arc::new(Mutex::new([0u8; 8]));

    let bar = ProgressBar::new(total_operations);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {per_sec}")
        .expect("Failed to create progress bar style")
        .progress_chars("█▉▊▋▌▍▎▏  "));
    bar.set_message("Cracking SHA-1...");

    println!(
        "\nStarting SHA-1 crack on GPU with {} Rust schedulers...",
        num_schedulers
    );
    println!("Total search space: {}\n", total_operations);

    let start_time = Instant::now();
    let mut handles = vec![];

    for id in 0..num_schedulers {
        let tar_bytes_clone = tar_bytes.clone();
        let index_clone = Arc::clone(&global_search_index);
        let found_clone = Arc::clone(&found_flag);
        let result_clone = Arc::clone(&found_result);
        let bar_clone = bar.clone();

        let handle = thread::spawn(move || {
            let mut result_buffer = [0u8; 8];
            loop {
                if found_clone.load(Ordering::Relaxed) {
                    break;
                }

                let start_index = index_clone.fetch_add(gpu_batch_size, Ordering::Relaxed);
                if start_index >= total_operations {
                    break;
                }

                let batch_size = (total_operations - start_index).min(gpu_batch_size);

                let ret = unsafe {
                    search_on_gpu(
                        start_index,
                        batch_size,
                        tar_bytes_clone.as_ptr(),
                        result_buffer.as_mut_ptr(),
                        max_threads_per_threadgroup as c_int,
                    )
                };

                bar_clone.inc(batch_size);

                if ret == 1 {
                    if found_clone
                        .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                        .is_ok()
                    {
                        let mut locked_result = result_clone.lock().unwrap();
                        locked_result.copy_from_slice(&result_buffer);
                        bar_clone.println(format!(
                            "\n[Scheduler {}] Match found! Stopping other threads...",
                            id
                        ));
                    }
                    break;
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    bar.finish_with_message("Search complete.");
    unsafe { cleanup_metal() };

    let duration = start_time.elapsed();
    let total_hashes_processed = global_search_index
        .load(Ordering::Relaxed)
        .min(total_operations);
    let hashes_per_second = total_hashes_processed as f64 / duration.as_secs_f64();

    println!("\n\n=== GPU Performance Statistics ===");
    println!("GPU: {}", gpu_name);
    println!("Total Time: {:.3?}", duration);
    println!("Total Hashes: {}", total_hashes_processed);
    println!("Hash Rate: {:.2} MH/s", hashes_per_second / 1_000_000.0);
    println!(
        "Rate Per Core: {:.2} MH/s",
        hashes_per_second / 1_000_000.0 / gpu_cores as f64
    );

    if found_flag.load(Ordering::Relaxed) {
        let result = found_result.lock().unwrap();
        let result_str = std::str::from_utf8(&*result).unwrap_or("invalid utf8");
        println!("\n🎉 Found Result: {}", result_str);
        println!("   Matching Hash: {}", tar_hex);
    } else {
        println!("\nResult not found in the search space.");
    }
}
