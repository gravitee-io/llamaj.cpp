# Devices & Memory

> Discover the GGML backends/devices llama.cpp registered, query CPU/GPU/RPC memory, read model dimensions, and inspect generation performance.

## Overview
These utilities let you introspect the runtime before and after a model is loaded: enumerate registered backends and devices (`BackendRegistry`), measure available memory on CPU RAM, the best local GPU, or remote RPC servers, cheaply read a GGUF model's dimensions without allocating its weights (`LlamaModelDims`), and read throughput/timing metrics from a running iterator (`LlamaPerformance`). Use them to drive capacity planning (will this model fit?), pick a device/`SplitMode`, or surface tokens/sec to users.

## Key types
- `BackendRegistry` — static facade: `listBackends()`, `listDevices()`, `supportsRpc()`, `loadBackend()`, `addRpcServer(s)()`, `queryRpcMemory()`, `printSummary()`.
- `GgmlBackendInfo` — record `(String name, long index)` for one registered backend (CPU, Metal, CUDA, RPC, ...).
- `GgmlDeviceInfo` — record `(String name, String description, long index, String backendName)` for one device.
- `CpuMemoryQuery` — `query()` returns `CpuMemoryInfo(freeBytes, totalBytes)` from the JDK MXBean (nullable, never throws).
- `GpuMemoryQuery` — `queryBest()` returns `GpuMemoryInfo(freeBytes, totalBytes)` for the GPU with most free VRAM (nullable).
- `RpcMemoryQuery` — `queryAll(List<String> endpoints)` returns aggregated `RpcMemoryInfo(freeBytes, totalBytes, serverCount)` (nullable).
- `LlamaModelDims` — record `(totalWeightBytes, nLayers, nHead, nHeadKv, headDim)`; `loadFrom(Path)` reads GGUF metadata without allocating weights.
- `LlamaPerformance` — `(ContextPerformance context, SamplerPerformance sampler)`; obtained via `LlamaIterator.getPerformance()`.
- `SplitMode` — `NONE`, `LAYER`, `ROW` (multi-GPU split strategy on `LlamaModelParams`).
- `AttentionType` — `UNSPECIFIED`, `CAUSAL`, `NON_CAUSAL` (on `LlamaContextParams`).

## Usage
```java
import io.gravitee.llama.cpp.*;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;

// 1. Initialise the runtime and register every backend in the lib directory.
String libPath = LlamaLibLoader.load();
LlamaRuntime.llama_backend_init();
try (Arena arena = Arena.ofConfined()) {
  LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

  // 2. Enumerate backends and devices.
  for (GgmlBackendInfo backend : BackendRegistry.listBackends()) {
    System.out.println(backend); // e.g. GgmlBackendInfo[name=Metal, index=1]
  }
  for (GgmlDeviceInfo device : BackendRegistry.listDevices()) {
    System.out.println(device);  // name, description, index, backend
  }
  BackendRegistry.printSummary(); // backends + devices + RPC/GPU offload flags

  // 3. Query memory: GPU first, fall back to system RAM.
  GpuMemoryQuery.GpuMemoryInfo gpu = GpuMemoryQuery.queryBest();
  if (gpu != null) {
    System.out.printf("GPU free=%d total=%d%n", gpu.freeBytes(), gpu.totalBytes());
  } else {
    CpuMemoryQuery.CpuMemoryInfo cpu = CpuMemoryQuery.query(); // null on failure
    System.out.printf("CPU free=%d total=%d%n", cpu.freeBytes(), cpu.totalBytes());
  }

  // 4. Estimate model footprint WITHOUT loading the weights (O(ms)).
  LlamaModelDims dims = LlamaModelDims.loadFrom(Path.of("/path/to/model.gguf"));
  System.out.printf(
    "weightBytes=%,d nLayers=%d nHead=%d nHeadKv=%d headDim=%d%n",
    dims.totalWeightBytes(), dims.nLayers(),
    dims.nHead(), dims.nHeadKv(), dims.headDim()
  );
}
LlamaRuntime.llama_backend_free();

// 5. After running an iterator, read throughput/timing metrics.
//    (iterator obtained from the text-generation path)
// LlamaPerformance perf = iterator.getPerformance();
// perf.generationTokensPerSecond();
// perf.promptTokensPerSecond();
// perf.totalProcessingTimeMs();
// perf.context().tokensGenerated();
// perf.sampler().averageSamplingTimeMs();
```

## Options

| Type / method | Values / shape | Notes |
|---|---|---|
| `SplitMode` | `NONE`, `LAYER`, `ROW` | `LlamaModelParams.splitMode(...)`; how layers/KV are spread across GPUs. |
| `AttentionType` | `UNSPECIFIED`, `CAUSAL`, `NON_CAUSAL` | `LlamaContextParams.attentionType(...)`; encoder vs decoder attention. |
| `GpuMemoryQuery.queryBest()` | `GpuMemoryInfo(freeBytes, totalBytes)` \| `null` | Picks the GPU/GPU_UMA device with the most free VRAM. |
| `CpuMemoryQuery.query()` | `CpuMemoryInfo(freeBytes, totalBytes)` \| `null` | JDK `OperatingSystemMXBean`; no native lib needed. |
| `RpcMemoryQuery.queryAll(endpoints)` | `RpcMemoryInfo(freeBytes, totalBytes, serverCount)` \| `null` | Sums free/total across de-duplicated `host:port` servers. |
| `BackendRegistry.queryRpcMemory(arena, endpoint, device)` | `RpcDeviceMemory(freeBytes, totalBytes)` | Single remote device; `usedBytes()` derived. |
| `LlamaPerformance.context()` | `ContextPerformance(startTimeMs, loadTimeMs, promptEvalTimeMs, evalTimeMs, promptTokensEvaluated, tokensGenerated, tokensReused)` | Timing + token counts. |
| `LlamaPerformance.sampler()` | `SamplerPerformance(samplingTimeMs, sampleCount)` | `averageSamplingTimeMs()` derived. |

## Notes
- `BackendRegistry`, `GpuMemoryQuery`, `RpcMemoryQuery`, and `LlamaModelDims` use FFM bindings: call `LlamaRuntime.llama_backend_init()` and `ggml_backend_load_all_from_path(...)` first, then `llama_backend_free()` at shutdown. `CpuMemoryQuery` is pure JDK and needs no native init.
- Memory queries return `null` on any failure (no GPU present, native lib not loaded, network error) and never throw — always null-check.
- `LlamaModelDims.loadFrom(Path)` loads with `noAlloc(true)`, `useExtraBufferTypes(false)`, `useMmap(false)`, `nGpuLayers(0)` and frees the model immediately, so it does not allocate weight bytes. It is safe to call even after all backends are registered (avoids the `GGML_ASSERT(!ml.no_alloc)` crash) and is idempotent.
- `addRpcServer`/`addRpcServers` perform a TCP pre-flight check and throw `LlamaException` on unreachable hosts; the remote `rpc-server` must match the client llama.cpp version or the native client will `abort()` the process. RPC devices are NOT in the global registry — pass the returned `MemorySegment` handles to `LlamaModelParams.devices(arena, list)`.
- `LlamaPerformance` is read from a live `LlamaIterator` via `getPerformance()`; derived getters return `0` when no prompt eval / generation / sampling occurred.
- `humanReadable` formatting (B/KiB/MiB/GiB/TiB) is built into `RpcDeviceMemory.toString()`; the memory records elsewhere expose raw bytes via `freeBytes()` / `totalBytes()`.
- All `Arena`-taking methods allocate in the supplied arena; use `Arena.ofConfined()` in a try-with-resources to bound native lifetime.

## See also
- [Distributed Inference (RPC)](../distributed-inference/README.md) — register and split a model across remote RPC servers.
- [Getting Started](../getting-started/README.md) — backend init, model/context setup, and the iterator that exposes `getPerformance()`.
- [Custom Builds & Platform Support](../custom-builds/README.md) — which backends (Metal, CUDA, CPU) get loaded per platform.
- [Quantized KV Cache](../quantized-kv-cache/README.md) — reduce context memory once you know the model dimensions.
