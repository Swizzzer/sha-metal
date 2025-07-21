# Sha-metal

基于macOS Metal库提供的并行计算能力，在GPU上执行sha1爆破。

## Usage

`sha1_rs <TARGET_HASH> <LENGTH_RANGE>`

**Arguments:**

  - <TARGET_HASH>   The target SHA-1 hash in hexadecimal format
  - <LENGTH_RANGE>  The length or length-range of the string to search