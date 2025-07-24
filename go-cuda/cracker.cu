#include "cracker.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "sha1_kernel.cu"


void bytes_to_uint32_be(const uint8_t *bytes, uint32_t *out, int count)
{
  for (int i = 0; i < count; ++i)
  {
    out[i] = ((uint32_t)bytes[i * 4] << 24) |
             ((uint32_t)bytes[i * 4 + 1] << 16) |
             ((uint32_t)bytes[i * 4 + 2] << 8) | (uint32_t)bytes[i * 4 + 3];
  }
}
extern "C" int initCuda(GPUInfo *info)
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0)
  {
    fprintf(stderr, "CUDA error: No CUDA-enabled devices found\n");
    return -1;
  }
  int dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  strncpy(info->name, deviceProp.name, 255);
  info->name[255] = '\0';
  info->multiProcessorCount = deviceProp.multiProcessorCount;
  info->cudaMajor = deviceProp.major;
  info->cudaMinor = deviceProp.minor;
  info->totalGlobalMem = deviceProp.totalGlobalMem;
  return 0;
}


extern "C" int searchOnGpu(const uint8_t* target_hash_bytes, uint64_t start_index, uint64_t count, char* result_str, int length, const char* alphabet, int alphabet_len) {
    if (length > MAX_CRACK_LENGTH) return -1;
    if (alphabet_len > MAX_ALPHABET_SIZE) return -1;

    cudaError_t err = cudaMemcpyToSymbol(d_alphabet, alphabet, alphabet_len * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    uint32_t target_hash_u32[5];
    bytes_to_uint32_be(target_hash_bytes, target_hash_u32, 5);
    uint32_t* d_target_hash;
    volatile uint32_t* d_found_flag;
    char* d_result_str;
    uint32_t h_found_flag = 0;
    err = cudaMalloc((void**)&d_target_hash, 5 * sizeof(uint32_t)); if (err != cudaSuccess) return -1;
    err = cudaMalloc((void**)&d_found_flag, sizeof(uint32_t)); if (err != cudaSuccess) return -1;
    err = cudaMalloc((void**)&d_result_str, (MAX_CRACK_LENGTH + 1) * sizeof(char)); if (err != cudaSuccess) return -1;
    err = cudaMemcpy(d_target_hash, target_hash_u32, 5 * sizeof(uint32_t), cudaMemcpyHostToDevice); if (err != cudaSuccess) return -1;
    err = cudaMemcpy((void*)d_found_flag, &h_found_flag, sizeof(uint32_t), cudaMemcpyHostToDevice); if (err != cudaSuccess) return -1;
    int threadsPerBlock = 256;
    dim3 blocksPerGrid((unsigned int)((count + threadsPerBlock - 1) / threadsPerBlock), 1, 1);
    int maxGridSize;
    cudaDeviceGetAttribute(&maxGridSize, cudaDevAttrMaxGridDimX, 0);
    if (blocksPerGrid.x > maxGridSize) { blocksPerGrid.x = maxGridSize; }
    sha1_search_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_target_hash, start_index, d_found_flag, d_result_str, length, alphabet_len);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(&h_found_flag, (const void*)d_found_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    int result = 0;
    if (h_found_flag != 0) {
        cudaMemcpy(result_str, d_result_str, (length + 1) * sizeof(char), cudaMemcpyDeviceToHost);
        result = 1;
    }
    cudaFree(d_target_hash);
    cudaFree((void*)d_found_flag);
    cudaFree(d_result_str);
    return result;
}

extern "C" void cleanupCuda() {
    cudaDeviceReset();
}