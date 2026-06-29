# Distributed Inference (RPC)

> Offload a model's layers and KV-cache to one or more remote `rpc-server` backends so inference runs across several machines.

## Overview
llama.cpp ships an `rpc-server` binary that exposes a machine's GPU/CPU as a network backend. llamaj.cpp lets you register those endpoints as devices and pin model offloading to them, so the heavy weights live on the remote nodes instead of (or in addition to) the local GPU. Use this when a model is too large for one machine, or to pool VRAM across a cluster. When several servers are registered with `splitMode=LAYER`, llama.cpp distributes layers proportionally to each server's free VRAM.

## Key types
- `BackendRegistry` — static helper to discover backends/devices and register RPC servers (`supportsRpc`, `addRpcServer`, `addRpcServers`, `queryRpcMemory`, `printSummary`).
- `BackendRegistry.RpcDeviceMemory` — record `(long freeBytes, long totalBytes)` with `usedBytes()`, for a single remote device.
- `RpcMemoryQuery` — pre-flight, registration-free aggregate VRAM check across endpoints; `queryAll(List<String>)` returns `null` on any failure (never throws).
- `RpcMemoryQuery.RpcMemoryInfo` — record `(long freeBytes, long totalBytes, int serverCount)` summed across all servers.
- `LlamaModelParams` — model load params; `rpcServers(...)`, `devices(...)`, `nGpuLayers(...)`, `splitMode(...)`.
- `SplitMode` — `NONE`, `LAYER`, `ROW` (how layers/KV are split across devices).

## Usage

Start one or more RPC server nodes first. The helper script downloads the matching prebuilt `rpc-server` and runs it (defaults to `0.0.0.0:50052`, Metal-only on macOS):

```bash
# On each remote machine (or another terminal)
./scripts/start-rpc-server.sh                       # 0.0.0.0:50052
./scripts/start-rpc-server.sh -H 127.0.0.1 -p 50053 # custom host/port
./scripts/start-rpc-server.sh -d CPU                # force CPU device
```

Then point the client at it. Registering an RPC server returns its device handles, which you pass to `devices(...)` so offloading targets only the remote nodes:

```java
import io.gravitee.llama.cpp.*;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;

public class RpcExample {
  public static void main(String[] args) {
    var arena = Arena.ofConfined();

    // Initialize runtime (loads native libs incl. the RPC backend plugin)
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();

    if (!BackendRegistry.supportsRpc()) {
      throw new IllegalStateException("RPC backend not available in this build");
    }

    // Optional pre-flight: check pooled free VRAM before loading the model
    var mem = RpcMemoryQuery.queryAll(java.util.List.of("127.0.0.1:50052"));
    if (mem != null) {
      System.out.printf("RPC pool: %d bytes free across %d server(s)%n",
        mem.freeBytes(), mem.serverCount());
    }

    // Register remote RPC servers -- returns their device handles
    var rpcDevices = BackendRegistry.addRpcServer(arena, "127.0.0.1:50052");

    // Print all discovered backends and devices (diagnostics)
    BackendRegistry.printSummary();

    // Load the model, restricting offloading to ONLY the RPC devices
    var modelParams = new LlamaModelParams(arena)
      .devices(arena, rpcDevices)
      .nGpuLayers(999);
    var model = new LlamaModel(arena, Path.of("models/model.gguf"), modelParams);

    // Everything else is identical to local inference
    var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512);
    var context = new LlamaContext(arena, model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).temperature(0.7f).seed(42);

    var state = ConversationState.create(arena, context, tokenizer, sampler, 0)
      .setMaxTokens(100)
      .initialize("What is the capital of France?");

    var iterator = new DefaultLlamaIterator(state);
    while (iterator.hasNext()) {
      System.out.print(iterator.next().text());
    }

    context.free();
    sampler.free();
    model.free();
    LlamaRuntime.llama_backend_free();
  }
}
```

Multiple servers: call `BackendRegistry.addRpcServers(arena, "192.168.1.10:50052", "192.168.1.11:50052")` (duplicate endpoints are ignored), or use `modelParams.rpcServers(arena, ...)` which registers them and lets llama.cpp split across all devices automatically. From the CLI, pass `--rpc host1:port,host2:port`.

## Options

| Knob | Where | Values / default | Effect |
| --- | --- | --- | --- |
| `splitMode(SplitMode)` | `LlamaModelParams` | `NONE`, `LAYER` (default for multi-device), `ROW` | How layers/KV split across devices; `LAYER` spreads proportionally to free VRAM, `ROW` uses tensor parallelism where supported |
| `nGpuLayers(int)` | `LlamaModelParams` | e.g. `999` | Layers to offload; use a large value to push all layers to the (remote) GPU devices |
| `devices(arena, List<MemorySegment>)` | `LlamaModelParams` | RPC device handles | Pins offloading to exactly these devices (e.g. only the remote ones, skipping local Metal) |
| `rpcServers(arena, String...)` | `LlamaModelParams` | `host:port` list | Registers servers and offloads across all devices automatically |
| `-v / -H / -p / -d` | `start-rpc-server.sh` | version / host / port / device (`MTL0`, `CPU`, ...) | Server bind + backend selection |

## Notes
- **Version must match.** The remote `rpc-server` must be the same llama.cpp build as the client library; a protocol mismatch makes the native RPC client `abort()` and kill the JVM. The bundled `start-rpc-server.sh` pins the matching release tag and caches the binary in `$HOME/.llama.cpp/rpc-server/` (delete that dir to force re-download).
- **Pre-flight TCP check.** `addRpcServer` opens a TCP socket (5s timeout) before calling native code and throws `LlamaException` on an unreachable host, avoiding an unrecoverable native crash. `RpcMemoryQuery.queryAll` instead returns `null` on any failure and never throws — prefer it for safe capacity checks before loading.
- **RPC devices are not global.** Devices returned by `addRpcServer`/`addRpcServers` are NOT in the global device registry; you must hand them to `devices(...)` (or rely on `rpcServers(...)`) for the model to use them. `getRpcDeviceHandles()` retrieves the accumulated handles.
- **Exclusive offload.** Using `devices(arena, rpcDevices)` loads weights only on the remote servers — the local GPU is not used.
- **Memory aggregation.** `RpcMemoryInfo.freeBytes`/`totalBytes` are the **sum** across servers and endpoints are deduplicated, since llama.cpp shares one registry entry per unique `host:port`.
- **Lifecycle.** Allocate native params from an `Arena` and `free()` `context`, `sampler`, and `model` (then `LlamaRuntime.llama_backend_free()`) when done; an `Arena.ofConfined()` releases everything it owns on close.
- **Ordering.** Register RPC servers after `llama_backend_init()` / `ggml_backend_load_all_from_path(...)` and before constructing the `LlamaModel`.

## See also
- [Devices & Memory](../device-and-memory/README.md) — device enumeration, VRAM estimation, and offload tuning.
- [Getting Started](../getting-started/README.md) — runtime init, model loading, and the basic generation loop.
- [Text Generation & Sampling](../text-generation/README.md) — the iterator/sampler API used after the model loads.
- [Custom Builds & Platform Support](../custom-builds/README.md) — ensuring the RPC backend plugin is present in your build.
