#ifndef CRACKER_H
#define CRACKER_H

#include <stdint.h>

#define MAX_CRACK_LENGTH 128
#define MAX_ALPHABET_SIZE 256

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[256];
    int multiProcessorCount;
    int cudaMajor;
    int cudaMinor;
    long long totalGlobalMem;
} GPUInfo;

int initCuda(GPUInfo* info);

int searchOnGpu(const uint8_t* target_hash, uint64_t start_index, uint64_t count, char* result_str, int length, const char* alphabet, int alphabet_len);

void cleanupCuda();

#ifdef __cplusplus
}
#endif

#endif