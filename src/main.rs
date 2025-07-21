use indicatif::{ProgressBar, ProgressStyle};
use metal::*;
use rayon::prelude::*;
use std::ffi::c_void;
use std::sync::atomic::{self, AtomicU32};
use std::time::Instant;

const CHARSET: &[u8] = b"0123456789abced";
const MSG_LEN: usize = 8;
const TARGET_HASH_STR: &str = "e0378e12d7ac5f9af37052d8763be4f3e8d13041"; // "abcdabcd"

// 最大支持的消息长度，用于内存布局。必须与 Metal 代码同步。
const MAX_MSG_LEN: usize = 8;

const BATCH_SIZE: usize = 1 << 22;
const SHADER_SRC: &str = include_str!("sha1.metal");

fn main() {
    assert!(
        MSG_LEN <= MAX_MSG_LEN,
        "MSG_LEN ({}) cannot exceed MAX_MSG_LEN ({})",
        MSG_LEN,
        MAX_MSG_LEN
    );

    let charset_len = CHARSET.len();
    let total_candidates = charset_len.pow(MSG_LEN as u32) as u64;

    println!("Target Hash: {}", TARGET_HASH_STR);
    println!("Charset: {}", String::from_utf8_lossy(CHARSET));
    println!("Message Length: {}", MSG_LEN);
    println!(
        "Total Search Space: {} ({} ^ {})",
        total_candidates, charset_len, MSG_LEN
    );
    println!("Batch Size: {} candidates", BATCH_SIZE);
    println!("Using GPU: {}", Device::system_default().unwrap().name());
    println!("---------------------------------------");

    let device = Device::system_default().expect("No Metal device found.");

    let library = device
        .new_library_with_source(SHADER_SRC, &CompileOptions::new())
        .expect("Failed to compile Metal library.");

    let kernel = library
        .get_function("sha1_kernel", None)
        .expect("Failed to get kernel function.");

    let pipeline_state = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Failed to create pipeline state.");

    let target_hash = parse_hash(TARGET_HASH_STR);

    let msgs_buffer = device.new_buffer(
        (BATCH_SIZE * MAX_MSG_LEN) as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    let found_buffer = device.new_buffer(
        std::mem::size_of::<AtomicU32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let target_hash_buffer = device.new_buffer_with_data(
        target_hash.as_ptr() as *const c_void,
        (target_hash.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    let msg_len_u32 = MSG_LEN as u32;

    let msg_len_buffer = device.new_buffer_with_data(
        &msg_len_u32 as *const u32 as *const c_void,
        4,
        MTLResourceOptions::StorageModeManaged,
    );

    let command_queue = device.new_command_queue();
    let start_time = Instant::now();
    let pb = ProgressBar::new(total_candidates);
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | {per_sec} | ETA: {eta}").unwrap().progress_chars("#>-"));

    let mut start: u64 = 0;
    while start < total_candidates {
        let current_batch_size = (total_candidates - start).min(BATCH_SIZE as u64) as usize;
        let msgs_vec = generate_msgs_for_batch(start, current_batch_size, CHARSET, MSG_LEN);
        let ptr = msgs_buffer.contents();

        unsafe {
            std::ptr::copy_nonoverlapping(msgs_vec.as_ptr(), ptr as *mut u8, msgs_vec.len());
        }

        msgs_buffer.did_modify_range(NSRange::new(0, msgs_vec.len() as u64));

        let found_ptr = found_buffer.contents() as *mut AtomicU32;

        unsafe {
            (*found_ptr).store(u32::MAX, atomic::Ordering::Relaxed);
        }

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&msgs_buffer), 0);
        encoder.set_buffer(1, Some(&found_buffer), 0);
        encoder.set_buffer(2, Some(&target_hash_buffer), 0);
        encoder.set_buffer(3, Some(&msg_len_buffer), 0);

        let grid_size = MTLSize {
            width: current_batch_size as u64,
            height: 1,
            depth: 1,
        };

        let threadgroup_size = pipeline_state
            .max_total_threads_per_threadgroup()
            .min(current_batch_size as u64);

        let threadgroup_dims = MTLSize {
            width: threadgroup_size,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_size, threadgroup_dims);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let found_idx = unsafe {
            (*(found_buffer.contents() as *const AtomicU32)).load(atomic::Ordering::Relaxed)
        };

        if found_idx != u32::MAX {
            let found_global_idx = start + found_idx as u64;
            let msg = index_to_msg(found_global_idx, CHARSET, MSG_LEN);
            pb.finish_and_clear();
            println!("\n✅ Success!");
            println!("   Message: {}", msg);
            println!("   Time taken: {:.2?}s", start_time.elapsed().as_secs_f32());
            let speed =
                (start + current_batch_size as u64) as f64 / start_time.elapsed().as_secs_f64();
            println!("   Average speed: {:.2} MHashes/sec", speed / 1_000_000.0);
            return;
        }

        start += current_batch_size as u64;
        pb.set_position(start);
    }

    pb.finish_and_clear();
    println!("\n❌ Message not found in the search space.");
}

fn parse_hash(hash_str: &str) -> [u32; 5] {
    let mut hash = [0u32; 5];
    for i in 0..5 {
        let sub = &hash_str[i * 8..(i + 1) * 8];
        hash[i] = u32::from_str_radix(sub, 16).expect("Invalid hash format");
    }
    hash
}

fn generate_msgs_for_batch(
    batch_start_idx: u64,
    batch_size: usize,
    charset: &[u8],
    msg_len: usize,
) -> Vec<u8> {
    let mut msgs = vec![0u8; batch_size * MAX_MSG_LEN];
    let charset_len = charset.len() as u64;

    msgs.par_chunks_mut(MAX_MSG_LEN)
        .enumerate() // 获取每个块的索引 (0, 1, 2...)
        .for_each(|(i, msg_chunk)| {
            let current_idx = batch_start_idx + i as u64;
            let mut temp_idx = current_idx;

            for j in 0..msg_len {
                let char_idx = (temp_idx % charset_len) as usize;
                msg_chunk[j] = charset[char_idx];
                temp_idx /= charset_len;
            }
        });

    msgs
}

fn index_to_msg(index: u64, charset: &[u8], msg_len: usize) -> String {
    if index == u64::MAX {
        return "<not found>".to_string();
    }
    let mut pass = vec![0u8; msg_len];
    let mut temp_idx = index;
    let charset_len = charset.len() as u64;
    for i in 0..msg_len {
        let char_idx = (temp_idx % charset_len) as usize;
        pass[i] = charset[char_idx];
        temp_idx /= charset_len;
    }
    String::from_utf8(pass).unwrap_or_else(|_| "<invalid utf8>".to_string())
}
