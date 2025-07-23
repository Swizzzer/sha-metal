# Sha-metal

基于macOS Metal库提供的并行计算能力，在GPU上执行sha1爆破。爆破字符限定为hex字符(0-9,a-e)

## Note

- 目前使用的字母表是固定的，暂不支持自定义字母表

## Usage

`sha1_rs <TARGET_HASH> <LENGTH_RANGE>`

**Arguments:**

  - `<TARGET_HASH>`   The target SHA-1 hash in hexadecimal format
  - `<LENGTH_RANGE>`  The length or length-range of the string to search
