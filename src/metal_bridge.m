#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

id<MTLDevice> device;
id<MTLCommandQueue> commandQueue;
id<MTLComputePipelineState> computePipelineState;
id<MTLBuffer> resultBuffer;
id<MTLBuffer> targetBuffer;
id<MTLBuffer> foundBuffer;
id<MTLBuffer> baseIndexBuffer;

const char *sha1MetalSource = R"(
#include <metal_stdlib>
using namespace metal;

inline uint32_t rotateLeft(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

inline uchar indexToChar(uint64_t index) {
    const uchar chars[16] = {
        '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
    };
    return chars[index & 0xF];
}

void sha1_hash_local(thread const uchar* input, thread uchar* output) {
    uint32_t H[5] = {
        0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    };
    uint32_t W[80];
    W[0] = ((uint32_t)input[0] << 24) | ((uint32_t)input[1] << 16) | ((uint32_t)input[2] << 8) | input[3];
    W[1] = ((uint32_t)input[4] << 24) | ((uint32_t)input[5] << 16) | ((uint32_t)input[6] << 8) | input[7];
    W[2] = 0x80000000;
    for(int i=3; i<14; ++i) W[i] = 0;
    W[14] = 0;
    W[15] = 64;

    for (int i = 16; i < 80; i++) {
        W[i] = rotateLeft(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1);
    }

    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4];
    uint32_t f, k;

    for (int i = 0; i < 80; i++) {
        if (i < 20) { f = (b & c) | ((~b) & d); k = 0x5A827999; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
        else { f = b ^ c ^ d; k = 0xCA62C1D6; }

        uint32_t temp = rotateLeft(a, 5) + f + e + k + W[i];
        e = d; d = c; c = rotateLeft(b, 30); b = a; a = temp;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e;

    for (int i = 0; i < 5; i++) {
        output[i*4 + 0] = (H[i] >> 24) & 0xff;
        output[i*4 + 1] = (H[i] >> 16) & 0xff;
        output[i*4 + 2] = (H[i] >> 8)  & 0xff;
        output[i*4 + 3] = H[i]         & 0xff;
    }
}

kernel void sha1_search(
    device uchar* result_out [[buffer(0)]],
    constant uchar* target [[buffer(1)]],
    device atomic_uint* found [[buffer(2)]],
    constant uint64_t* baseIndex [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (atomic_load_explicit(found, memory_order_relaxed) != 0) return;

    uint64_t candidateIndex = baseIndex[0] + gid;
    thread uchar candidate[8];
    uint64_t idx = candidateIndex;
    for (int i = 7; i >= 0; i--) {
        candidate[i] = indexToChar(idx);
        idx >>= 4;
    }

    thread uchar hash[20];
    sha1_hash_local(candidate, hash);

    bool match = true;
    for (int i = 0; i < 20; i++) {
        if (hash[i] != target[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        atomic_store_explicit(found, 1, memory_order_relaxed);
        for (int i = 0; i < 8; i++) {
            result_out[i] = candidate[i];
        }
    }
}
)";

typedef struct {
  int core_count;
  int max_threads_per_threadgroup;
  char name[256];
} GPUInfo;

int get_gpu_cores_from_system_profiler() {
  FILE *fp;
  char buffer[128];
  int cores = 0;
  fp = popen("system_profiler SPDisplaysDataType | awk '/Total Number of "
             "Cores:/{print $5}'",
             "r");
  if (fp == NULL) {
    return 0;
  }
  if (fgets(buffer, sizeof(buffer), fp) != NULL) {
    cores = atoi(buffer);
  }
  pclose(fp);
  return cores;
}

__attribute__((visibility("default"))) int init_metal(GPUInfo *gpu_info) {
  @autoreleasepool {
    NSError *error = nil;
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return -1;
    }

    strncpy(gpu_info->name, [[device name] UTF8String], 255);
    gpu_info->name[255] = '\0';
    gpu_info->core_count = get_gpu_cores_from_system_profiler();
    if (gpu_info->core_count == 0) {
      gpu_info->core_count = 8;
    } // Fallback

    commandQueue = [device newCommandQueue];
    if (!commandQueue) {
      return -1;
    }

    NSString *source = [NSString stringWithUTF8String:sha1MetalSource];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = YES;

    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:options
                                                    error:&error];
    if (!library) {
      fprintf(stderr, "Failed to compile shader: %s\n",
              [[error description] UTF8String]);
      return -1;
    }

    id<MTLFunction> kernelFunction =
        [library newFunctionWithName:@"sha1_search"];
    if (!kernelFunction) {
      return -1;
    }

    computePipelineState =
        [device newComputePipelineStateWithFunction:kernelFunction
                                              error:&error];
    if (!computePipelineState) {
      fprintf(stderr, "Failed to create pipeline state: %s\n",
              [[error description] UTF8String]);
      return -1;
    }

    gpu_info->max_threads_per_threadgroup =
        (int)computePipelineState.maxTotalThreadsPerThreadgroup;

    resultBuffer = [device newBufferWithLength:8
                                       options:MTLResourceStorageModeShared];
    targetBuffer = [device newBufferWithLength:20
                                       options:MTLResourceStorageModeShared];
    foundBuffer = [device newBufferWithLength:sizeof(unsigned int)
                                      options:MTLResourceStorageModeShared];
    baseIndexBuffer = [device newBufferWithLength:sizeof(uint64_t)
                                          options:MTLResourceStorageModeShared];

    if (!resultBuffer || !targetBuffer || !foundBuffer || !baseIndexBuffer) {
      return -1;
    }
    return 0;
  }
}

__attribute__((visibility("default"))) int
search_on_gpu(uint64_t start_index, uint64_t count, const uint8_t *target,
              uint8_t *result, int max_threads_per_threadgroup) {
  @autoreleasepool {
    memcpy([targetBuffer contents], target, 20);
    *(uint64_t *)[baseIndexBuffer contents] = start_index;
    *(unsigned int *)[foundBuffer contents] = 0;

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:computePipelineState];
    [encoder setBuffer:resultBuffer offset:0 atIndex:0];
    [encoder setBuffer:targetBuffer offset:0 atIndex:1];
    [encoder setBuffer:foundBuffer offset:0 atIndex:2];
    [encoder setBuffer:baseIndexBuffer offset:0 atIndex:3];

    NSUInteger w = computePipelineState.threadExecutionWidth;
    NSUInteger threadsPerGroup = (max_threads_per_threadgroup / w) * w;
    if (threadsPerGroup == 0)
      threadsPerGroup = w;
    if (threadsPerGroup > computePipelineState.maxTotalThreadsPerThreadgroup) {
      threadsPerGroup = computePipelineState.maxTotalThreadsPerThreadgroup;
    }

    MTLSize threadsPerThreadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize grid = MTLSizeMake(count, 1, 1);

    [encoder dispatchThreads:grid
        threadsPerThreadgroup:threadsPerThreadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if (*(unsigned int *)[foundBuffer contents] != 0) {
      memcpy(result, [resultBuffer contents], 8);
      return 1;
    }
    return 0;
  }
}

__attribute__((visibility("default"))) void cleanup_metal() {
  device = nil;
  commandQueue = nil;
  computePipelineState = nil;
  resultBuffer = nil;
  targetBuffer = nil;
  foundBuffer = nil;
  baseIndexBuffer = nil;
}
