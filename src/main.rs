use indicatif::{ProgressBar, ProgressStyle};
use metal::*;
use std::ffi::c_void;
use std::sync::atomic::{self, AtomicU32};
use std::time::Instant;

const CHARSET: &[u8] = b"12345678ab";
const MSG_LEN: usize = 10;
const TARGET_HASH_STR: &str = "d36ecb3306c30fe434a768c9d8b6f4bf2eb9abb3"; // sha1(b"12345678ab")
const MAX_MSG_LEN: usize = 16;
const BATCH_SIZE: usize = 1 << 25;
const SHADER_SRC: &str = include_str!("sha1.metal");

fn main() {
    assert!(MSG_LEN <= MAX_MSG_LEN, "MSG_LEN cannot exceed MAX_MSG_LEN");

    let charset_len = CHARSET.len() as u64;
    let total_candidates = charset_len.pow(MSG_LEN as u32);
    println!("Target Hash: {}", TARGET_HASH_STR);
    println!("Message Length: {}", MSG_LEN);
    println!("Total Search Space: {}", total_candidates);
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

    // 交给GPU生成消息候选
    let found_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let target_hash_buffer = device.new_buffer_with_data(
        target_hash.as_ptr() as *const c_void,
        20,
        MTLResourceOptions::StorageModeManaged,
    );

    let msg_len_u32 = MSG_LEN as u32;
    let msg_len_buffer = device.new_buffer_with_data(
        &msg_len_u32 as *const u32 as *const c_void,
        4,
        MTLResourceOptions::StorageModeManaged,
    );

    let charset_buffer = device.new_buffer_with_data(
        CHARSET.as_ptr() as *const c_void,
        CHARSET.len() as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let charset_len_u64 = charset_len;
    let charset_len_buffer = device.new_buffer_with_data(
        &charset_len_u64 as *const u64 as *const c_void,
        8,
        MTLResourceOptions::StorageModeManaged,
    );

    let start_index_buffer = device.new_buffer(8, MTLResourceOptions::StorageModeManaged);

    let command_queue = device.new_command_queue();
    let start_time = Instant::now();
    let pb = ProgressBar::new(total_candidates);
    pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | {per_sec} | ETA: {eta}").unwrap().progress_chars("#>-"));

    let mut start: u64 = 0;
    while start < total_candidates {
        let current_batch_size = (total_candidates - start).min(BATCH_SIZE as u64) as usize;

        // 更新下一个起始索引
        let ptr = start_index_buffer.contents() as *mut u64;
        unsafe {
            *ptr = start;
        }
        start_index_buffer.did_modify_range(NSRange::new(0, 8));

        let found_ptr = found_buffer.contents() as *mut AtomicU32;
        unsafe {
            (*found_ptr).store(u32::MAX, atomic::Ordering::Relaxed);
        }

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&found_buffer), 0);
        encoder.set_buffer(1, Some(&target_hash_buffer), 0);
        encoder.set_buffer(2, Some(&msg_len_buffer), 0);
        encoder.set_buffer(3, Some(&charset_buffer), 0);
        encoder.set_buffer(4, Some(&charset_len_buffer), 0);
        encoder.set_buffer(5, Some(&start_index_buffer), 0);

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
    println!("   Time taken: {:.2?}s", start_time.elapsed().as_secs_f32());
}

fn parse_hash(hash_str: &str) -> [u32; 5] {
    let mut hash = [0u32; 5];
    for i in 0..5 {
        let sub = &hash_str[i * 8..(i + 1) * 8];
        hash[i] = u32::from_str_radix(sub, 16).expect("Invalid hash format");
    }
    hash
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
