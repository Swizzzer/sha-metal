package main

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -lsha1cracker
#include "cracker.h"
*/
import "C"
import (
	"context"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/schollz/progressbar/v3"
)

var (
	gpuInfo        C.GPUInfo
	GPUBatchSize   uint64
	foundFlag      atomic.Int32
	foundResult    [C.MAX_CRACK_LENGTH + 1]byte
	resultMutex    sync.Mutex
	globalProgress atomic.Uint64
)

var (
	targetHash = flag.String("hash", "", "Target SHA-1 hash to crack (required)")
	minLength  = flag.Int("min", 1, "Minimum length of the string to crack")
	maxLength  = flag.Int("max", 8, "Maximum length of the string to crack")
	alphabet   = flag.String("alphabet", "0123456789abcdef", "Custom alphabet for cracking")
)

func init() {
	fmt.Println("初始化 CUDA GPU...")
	if ret := C.initCuda(&gpuInfo); ret != 0 {
		log.Fatalf("CUDA 初始化失败")
	}

	gpuCores := int(gpuInfo.multiProcessorCount)
	GPUBatchSize = uint64(gpuCores * 256 * 256)
	if GPUBatchSize > (1 << 26) {
		GPUBatchSize = 1 << 26
	}
}

func main() {
	flag.Parse()

	if *targetHash == "" {
		fmt.Fprintln(os.Stderr, "错误: 必须提供 -hash 参数。")
		flag.Usage()
		os.Exit(1)
	}
	if len(*targetHash) != 40 {
		fmt.Fprintf(os.Stderr, "错误: 无效的 SHA-1 哈希长度。需要 40 个十六进制字符，但提供了 %d 个。\n", len(*targetHash))
		os.Exit(1)
	}
	tarBytes, err := hex.DecodeString(*targetHash)
	if err != nil {
		log.Fatalf("无法解码目标哈希: %v", err)
	}

	if len(*alphabet) > C.MAX_ALPHABET_SIZE {
		log.Fatalf("错误: 字母表长度 (%d) 不能超过编译时设定的最大值 (%d)", len(*alphabet), C.MAX_ALPHABET_SIZE)
	}
	if *maxLength > C.MAX_CRACK_LENGTH {
		log.Fatalf("错误: 最大长度 (%d) 不能超过编译时设定的最大值 (%d)", *maxLength, C.MAX_CRACK_LENGTH)
	}

	fmt.Printf("\n=== GPU Information ===\n")
	fmt.Printf("GPU: %s\n", C.GoString(&gpuInfo.name[0]))
	fmt.Printf("Streaming Multiprocessors (SMs): %d\n", gpuInfo.multiProcessorCount)
	fmt.Printf("\n=== 配置 ===\n")
	fmt.Printf("目标哈希: %s\n", *targetHash)
	fmt.Printf("搜索范围: 长度 %d 到 %d\n", *minLength, *maxLength)
	fmt.Printf("使用字母表: \"%s\" (基数 %d)\n", *alphabet, len(*alphabet))
	fmt.Printf("批处理大小: %d (%.2fM hashes)\n", GPUBatchSize, float64(GPUBatchSize)/(1024*1024))
	fmt.Println("\nCUDA GPU 初始化成功！")

	runtime.GOMAXPROCS(runtime.NumCPU())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	timeStart := time.Now()
	alphabetBytes := []byte(*alphabet)
	alphabetLen := len(alphabetBytes)

	for length := *minLength; length <= *maxLength && foundFlag.Load() == 0; length++ {
		totalOperations := uint64(math.Pow(float64(alphabetLen), float64(length)))
		fmt.Printf("\n--- 开始搜索长度为 %d 的字符串 (共 %d 种可能) ---\n", length, totalOperations)

		bar := progressbar.NewOptions64(int64(totalOperations),
			progressbar.OptionSetDescription(fmt.Sprintf("Cracking len %d...", length)),
			progressbar.OptionShowBytes(false),
			progressbar.OptionSetWidth(30),
			progressbar.OptionShowCount(),
			progressbar.OptionSetTheme(progressbar.Theme{
				Saucer: "=", SaucerHead: ">", SaucerPadding: " ", BarStart: "[", BarEnd: "]",
			}),
			progressbar.OptionThrottle(100*time.Millisecond),
		)

		jobs := make(chan uint64, runtime.NumCPU())
		wg := &sync.WaitGroup{}
		globalProgress.Store(0)
		progressCtx, progressCancel := context.WithCancel(ctx)
		go progressUpdater(bar, progressCtx)

		for i := 0; i < runtime.NumCPU(); i++ {
			wg.Add(1)
			go gpuWorker(wg, jobs, length, alphabetBytes, tarBytes)
		}

		for i := uint64(0); i < totalOperations && foundFlag.Load() == 0; i += GPUBatchSize {
			select {
			case jobs <- i:
			case <-ctx.Done():
				break
			}
		}
		close(jobs)
		wg.Wait()
		progressCancel()
		bar.Finish()

		if foundFlag.Load() != 0 {
			cancel()
		}
	}

	C.cleanupCuda()
	timeEnd := time.Now()
	duration := timeEnd.Sub(timeStart)

	fmt.Printf("\n=== 性能统计 ===\n")
	fmt.Printf("GPU: %s\n", C.GoString(&gpuInfo.name[0]))
	fmt.Printf("总耗时: %v\n", duration)

	if foundFlag.Load() != 0 {
		resultMutex.Lock()
		res := foundResult
		resultMutex.Unlock()
		foundStr := C.GoString((*C.char)(unsafe.Pointer(&res[0])))
		fmt.Printf("\n🎉 找到结果! 🎉\n")
		fmt.Printf("原文: %s\n", foundStr)
		fmt.Printf("SHA1: %s\n", strings.ToLower(*targetHash))
	} else {
		fmt.Printf("\n在指定范围内未找到哈希 %s 的结果。\n", *targetHash)
	}
}

func progressUpdater(bar *progressbar.ProgressBar, ctx context.Context) {
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			bar.Set64(int64(globalProgress.Load()))
		case <-ctx.Done():
			bar.Set64(int64(globalProgress.Load()))
			return
		}
	}
}

func gpuWorker(wg *sync.WaitGroup, jobs <-chan uint64, length int, alphabet []byte, targetHash []byte) {
	defer wg.Done()
	result := make([]byte, C.MAX_CRACK_LENGTH+1)

	for startIndex := range jobs {
		if foundFlag.Load() != 0 {
			return
		}

		ret := C.searchOnGpu(
			(*C.uint8_t)(unsafe.Pointer(&targetHash[0])),
			C.uint64_t(startIndex),
			C.uint64_t(GPUBatchSize),
			(*C.char)(unsafe.Pointer(&result[0])),
			C.int(length),
			(*C.char)(unsafe.Pointer(&alphabet[0])),
			C.int(len(alphabet)),
		)

		globalProgress.Add(GPUBatchSize)

		if ret == 1 {
			if foundFlag.CompareAndSwap(0, 1) {
				resultMutex.Lock()
				copy(foundResult[:], result)
				resultMutex.Unlock()
			}
			return
		}
	}
}
