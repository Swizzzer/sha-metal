# SHA1-GPU-Cracking

利用 GPU 的并行计算能力，在 GPU 上高效执行 SHA-1 哈希爆破。

## 构建与使用方法

### 使用 Go 实现版本

确保系统已安装以下组件：`cmake`、`go` 和 NVIDIA CUDA。

```bash
mkdir build && cd build
cmake ..
make
LD_LIBRARY_PATH=. ./executable -min=[最小长度] -max=[最大长度] -alphabet=[候选字符集] -hash=[目标 SHA-1 哈希（十六进制）]
```

参数说明：

* `-min`：爆破字符串的最小长度
* `-max`：爆破字符串的最大长度
* `-alphabet`：用于构造候选字符串的字符集（如：abc123）
* `-hash`：目标 SHA-1 哈希值（16 进制字符串）

### 使用 Rust 实现版本

> ⚠️ 当前仅支持 Apple M 系列芯片。

```bash
cargo build --release
./target/release/sha1_rs <目标哈希> <长度范围>
```

示例：

```bash
./target/release/sha1_rs e005107414ab1b52bd43e6a294da0b5124a80a2c 8-10
```

