#include <metal_stdlib>
using namespace metal;

#define TO_BIG_ENDIAN(n) ( ((n) << 24) | (((n) << 8) & 0x00ff0000) | (((n) >> 8) & 0x0000ff00) | ((n) >> 24) )
inline uint rotl(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

constant constexpr uint MAX_MSG_LEN = 16;

kernel void sha1_kernel(
    device atomic_uint* found_index     [[buffer(0)]],
    constant const uint* target_hash    [[buffer(1)]],
    constant const uint& msg_len        [[buffer(2)]],
    constant const char* charset        [[buffer(3)]],
    constant const ulong& charset_len   [[buffer(4)]],
    constant const ulong& start_index   [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong global_index = start_index + gid;
    ulong temp_idx = global_index;
    
    uint block[16] = {0};
    thread uchar* p_block = (thread uchar*)block;

    for (uint i = 0; i < msg_len; ++i) {
        uint char_idx = temp_idx % charset_len;
        p_block[i] = charset[char_idx];
        temp_idx /= charset_len;
    }


    p_block[msg_len] = 0x80;    
    uint64_t bit_length = (uint64_t)msg_len * 8;
    p_block[56] = (uchar)(bit_length >> 56); p_block[57] = (uchar)(bit_length >> 48);
    p_block[58] = (uchar)(bit_length >> 40); p_block[59] = (uchar)(bit_length >> 32);
    p_block[60] = (uchar)(bit_length >> 24); p_block[61] = (uchar)(bit_length >> 16);
    p_block[62] = (uchar)(bit_length >> 8);  p_block[63] = (uchar)(bit_length);

    for (int i = 0; i < 16; ++i) {
        block[i] = TO_BIG_ENDIAN(block[i]);
    }

    uint h[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    uint w[80];
    for (int i = 0; i < 16; ++i) w[i] = block[i];
    for (int i = 16; i < 80; ++i) w[i] = rotl(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    uint a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
        uint f, k;
        if (i < 20) { f = (b & c) | ((~b) & d); k = 0x5A827999; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
        else { f = b ^ c ^ d; k = 0xCA62C1D6; }
        uint temp = rotl(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rotl(b, 30); b = a; a = temp;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    
    if (h[0] == target_hash[0] && h[1] == target_hash[1] && h[2] == target_hash[2] && h[3] == target_hash[3] && h[4] == target_hash[4]) {
        atomic_store_explicit(found_index, gid, memory_order_relaxed);
    }
}
