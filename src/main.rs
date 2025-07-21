use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
const MAX_CANDIDATE_LENGTH: usize = 16;

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
        length: u32,
    ) -> c_int;
    fn reset_found_flag();
    fn cleanup_metal();
}

#[derive(Parser, Debug)]
#[command(author, version="0.1.0", about="A simple tool to crack SHA-1 (msg less than 55bytes) with Metal", long_about = None)]
struct Args {
    #[arg(help = "The target SHA-1 hash in hexadecimal format")]
    target_hash: String,

    #[arg(help = "The length or length-range of the string to search, e.g., '8' or '4-8'")]
    length_range: String,
}

fn parse_length_range(range_str: &str) -> Result<(u32, u32), String> {
    if let Some((start_str, end_str)) = range_str.split_once('-') {
        let start = start_str
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("Invalid start of range: '{}'", start_str))?;
        let end = end_str
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("Invalid end of range: '{}'", end_str))?;
        if start == 0 || end == 0 {
            return Err("Length cannot be zero.".to_string());
        }
        if start > end {
            return Err(format!(
                "Start of range ({}) cannot be greater than end ({})",
                start, end
            ));
        }
        if end as usize > MAX_CANDIDATE_LENGTH {
            return Err(format!(
                "Maximum search length cannot exceed {}",
                MAX_CANDIDATE_LENGTH
            ));
        }
        Ok((start, end))
    } else {
        let len = range_str
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("Invalid length: '{}'", range_str))?;
        if len == 0 {
            return Err("Length cannot be zero.".to_string());
        }
        if len as usize > MAX_CANDIDATE_LENGTH {
            return Err(format!(
                "Search length cannot exceed {}",
                MAX_CANDIDATE_LENGTH
            ));
        }
        Ok((len, len))
    }
}

fn main() {
    let args = Args::parse();

    let (start_len, end_len) = parse_length_range(&args.length_range).unwrap_or_else(|err| {
        eprintln!("Error parsing length range: {}", err);
        std::process::exit(1);
    });

    let tar_bytes = hex::decode(&args.target_hash).unwrap_or_else(|_| {
        eprintln!("Error: Invalid hexadecimal string for target hash.");
        std::process::exit(1);
    });

    if tar_bytes.len() != 20 {
        eprintln!("Error: SHA-1 hash must be 20 bytes long (40 hex characters).");
        std::process::exit(1);
    }

    println!("Initializing Metal GPU...");
    let mut gpu_info = GPUInfo::default();
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
    println!("Target Hash: {}", args.target_hash);
    println!("Search Lengths: {}-{}", start_len, end_len);
    println!(
        "Batch Size: {} ({:.2}M hashes)",
        gpu_batch_size,
        gpu_batch_size as f64 / 1_048_576.0
    );

    let num_schedulers = num_cpus::get().min(4);
    let global_search_index = Arc::new(AtomicU64::new(0));
    let found_flag = Arc::new(AtomicBool::new(false));
    let found_result = Arc::new(Mutex::new(vec![0u8; MAX_CANDIDATE_LENGTH]));
    let found_length = Arc::new(Mutex::new(0u32));
    let start_time = Instant::now();

    for length in start_len..=end_len {
        if found_flag.load(Ordering::Relaxed) {
            break;
        }

        let current_search_space = 16u64.pow(length);
        println!(
            "\n--- Searching for length {} strings (Space: {}) ---",
            length, current_search_space
        );

        // ... (循环内部的逻辑与之前完全相同) ...
        global_search_index.store(0, Ordering::Relaxed);
        unsafe {
            reset_found_flag();
        }

        let bar = ProgressBar::new(current_search_space);
        bar.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {per_sec}")
            .expect("Failed to create progress bar style")
            .progress_chars("█▉▊▋▌▍▎▏  "));
        bar.set_message(format!("Cracking len {}", length));

        let mut handles = vec![];

        for id in 0..num_schedulers {
            // ... (线程创建和调度逻辑完全相同) ...
            let tar_bytes_clone = tar_bytes.clone();
            let index_clone = Arc::clone(&global_search_index);
            let found_clone = Arc::clone(&found_flag);
            let result_clone = Arc::clone(&found_result);
            let length_clone = Arc::clone(&found_length);
            let bar_clone = bar.clone();

            let handle = thread::spawn(move || {
                let mut result_buffer = vec![0u8; MAX_CANDIDATE_LENGTH];
                loop {
                    if found_clone.load(Ordering::Relaxed) {
                        break;
                    }
                    let start_index = index_clone.fetch_add(gpu_batch_size, Ordering::Relaxed);
                    if start_index >= current_search_space {
                        break;
                    }
                    let batch_size = (current_search_space - start_index).min(gpu_batch_size);
                    let ret = unsafe {
                        search_on_gpu(
                            start_index,
                            batch_size,
                            tar_bytes_clone.as_ptr(),
                            result_buffer.as_mut_ptr(),
                            max_threads_per_threadgroup as c_int,
                            length,
                        )
                    };
                    bar_clone.inc(batch_size);
                    if ret == 1 {
                        if found_clone
                            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                            .is_ok()
                        {
                            let mut locked_result = result_clone.lock().unwrap();
                            let mut locked_len = length_clone.lock().unwrap();
                            *locked_len = length;
                            locked_result[..length as usize]
                                .copy_from_slice(&result_buffer[..length as usize]);
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
        bar.finish_with_message(format!("Length {} search complete.", length));
    }

    unsafe { cleanup_metal() };

    // --- 3. 更新报告部分 ---
    let duration = start_time.elapsed();
    println!("\n\n=== Final Statistics ===");
    println!("Total Time: {:.3?}", duration);

    if found_flag.load(Ordering::Relaxed) {
        let result_vec = found_result.lock().unwrap();
        let len = *found_length.lock().unwrap() as usize;
        let result_str = std::str::from_utf8(&result_vec[..len]).unwrap_or("invalid utf8");
        println!("\n🎉 Found Result: {}", result_str);
        println!("   Length: {}", len);
        println!("   Matching Hash: {}", args.target_hash);
    } else {
        println!(
            "\nResult not found for lengths {} to {}.",
            start_len, end_len
        );
    }
}
