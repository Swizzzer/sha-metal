#include "cracker.h"
#include <cuda_runtime.h>
#include <stdint.h>

__constant__ char d_alphabet[MAX_ALPHABET_SIZE];

__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) { return (x << n) | (x >> (32 - n)); }
__device__ void sha1_transform(uint32_t state[5], const uint8_t block[64])
{
  uint32_t w[80];
  for (int i = 0; i < 16; ++i)
  {
    uint32_t temp = 0;
    temp |= (uint32_t)block[i * 4 + 0] << 24;
    temp |= (uint32_t)block[i * 4 + 1] << 16;
    temp |= (uint32_t)block[i * 4 + 2] << 8;
    temp |= (uint32_t)block[i * 4 + 3] << 0;
    w[i] = temp;
  }
  for (int i = 16; i < 80; ++i)
  {
    w[i] = rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
  }
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];
  for (int i = 0; i < 80; ++i)
  {
    uint32_t f, k;
    if (i < 20)
    {
      f = (b & c) | ((~b) & d);
      k = 0x5A827999;
    }
    else if (i < 40)
    {
      f = b ^ c ^ d;
      k = 0x6ED9EBA1;
    }
    else if (i < 60)
    {
      f = (b & c) | (b & d) | (c & d);
      k = 0x8F1BBCDC;
    }
    else
    {
      f = b ^ c ^ d;
      k = 0xCA62C1D6;
    }
    uint32_t temp = rotl32(a, 5) + f + e + k + w[i];
    e = d;
    d = c;
    c = rotl32(b, 30);
    b = a;
    a = temp;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
}

__global__ void sha1_search_kernel(
    const uint32_t *target_hash,
    uint64_t base_index,
    volatile uint32_t *found_flag,
    char *result_str,
    int length,
    int alphabet_len)
{
  uint64_t global_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t candidate_index = base_index + global_id;

  if (*found_flag != 0)
  {
    return;
  }

  uint8_t msg[MAX_CRACK_LENGTH];
  uint64_t temp_idx = candidate_index;
#pragma unroll
  for (int i = length - 1; i >= 0; --i)
  {
    msg[i] = d_alphabet[temp_idx % alphabet_len];
    temp_idx /= alphabet_len;
  }

  uint32_t state[5];
  state[0] = 0x67452301;
  state[1] = 0xEFCDAB89;
  state[2] = 0x98BADCFE;
  state[3] = 0x10325476;
  state[4] = 0xC3D2E1F0;
  uint8_t block[64];
  int processed_len = 0;
  while (processed_len + 64 <= length)
  {
    for (int i = 0; i < 64; ++i)
    {
      block[i] = msg[processed_len + i];
    }
    sha1_transform(state, block);
    processed_len += 64;
  }
  int remaining_len = length - processed_len;
  for (int i = 0; i < remaining_len; ++i)
  {
    block[i] = msg[processed_len + i];
  }
  block[remaining_len] = 0x80;

  if (remaining_len < 56)
  {
    for (int i = remaining_len + 1; i < 56; ++i)
    {
      block[i] = 0;
    }
  }
  else
  {
    for (int i = remaining_len + 1; i < 64; ++i)
    {
      block[i] = 0;
    }
    sha1_transform(state, block);
    for (int i = 0; i < 56; ++i)
    {
      block[i] = 0;
    }
  }
  uint64_t bit_len = (uint64_t)length * 8;
  block[56] = (uint8_t)(bit_len >> 56);
  block[57] = (uint8_t)(bit_len >> 48);
  block[58] = (uint8_t)(bit_len >> 40);
  block[59] = (uint8_t)(bit_len >> 32);
  block[60] = (uint8_t)(bit_len >> 24);
  block[61] = (uint8_t)(bit_len >> 16);
  block[62] = (uint8_t)(bit_len >> 8);
  block[63] = (uint8_t)(bit_len >> 0);

  sha1_transform(state, block);
  if (state[0] == target_hash[0] && state[1] == target_hash[1] &&
      state[2] == target_hash[2] && state[3] == target_hash[3] &&
      state[4] == target_hash[4])
  {
    if (atomicCAS((unsigned int *)found_flag, 0, 1) == 0)
    {
      for (int i = 0; i < length; ++i)
      {
        result_str[i] = msg[i];
      }
      if (length < MAX_CRACK_LENGTH)
      {
        result_str[length] = '\0';
      }
    }
  }
}