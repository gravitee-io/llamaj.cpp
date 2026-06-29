# Getting Started

> Initialize the native runtime, load a GGUF model, build an inference context, and tokenize text.

## Overview
llamaj.cpp is a Java FFM (Project Panama) binding over llama.cpp. Every program follows the
same bootstrap sequence: initialize the GGML backends, load the model weights, create a context
that holds the KV-cache and runtime state, then derive a vocab and tokenizer from the model.
This page covers that lifecycle and the `Arena`/`Freeable` resource-management model that keeps
native memory from leaking. Read it first — all other capabilities build on these types.

## Key types
- `LlamaRuntime` — static entry points to the native library: `llama_backend_init()`, `ggml_backend_load_all_from_path()`, `llama_backend_free()`.
- `LlamaLibLoader` — resolves and `System.load()`s the bundled (or `LLAMA_CPP_LIB_PATH`) native libs; `load()` returns the directory path to feed to `ggml_backend_load_all_from_path`.
- `LlamaModelParams` — builder over `llama_model_default_params` (GPU layers, mmap, mlock, vocab-only, ...).
- `LlamaModel` — a loaded GGUF model; `Freeable`. Exposes metadata helpers (`desc`, `metaVal`, `meta`, `nCtxTrain`, `nEmbdOut`).
- `LlamaContextParams` — builder over `llama_context_default_params` (`nCtx`, `nBatch`, `nUBatch`, `nSeqMax`, threads, ...).
- `LlamaContext` — the inference session bound to a model; `Freeable`. Owns the KV-cache.
- `LlamaVocab` — token <-> piece access derived from the model.
- `LlamaTokenizer` — encodes a `String` into token ids (`tokenize`) and decodes pieces (`tokenToPiece`).
- `Freeable` — interface with `free()`; all native-owning handles implement it and must be freed.
- `BackendRegistry` — optional high-level helper to enumerate backends/devices after init.

## Usage
```java
import io.gravitee.llama.cpp.*;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;

public class GettingStarted {
    public static void main(String[] args) {
        // 1. One Arena owns all the native param/segment allocations.
        var arena = Arena.ofConfined();

        // 2. Load the native libraries, then initialize and register the backends.
        String libPath = LlamaLibLoader.load();          // returns the native lib directory
        LlamaRuntime.llama_backend_init();
        LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

        // 3. Load the model. Builder methods return `this` for chaining.
        var modelParams = new LlamaModelParams(arena)
            .nGpuLayers(99)        // offload all layers to GPU (0 = CPU only)
            .useMmap(true);
        var model = new LlamaModel(arena, Path.of("models/model.gguf"), modelParams);

        // 4. Build a context (KV-cache + runtime state) from the model.
        var contextParams = new LlamaContextParams(arena)
            .nCtx(2048)            // context window in tokens
            .nBatch(512)           // logical batch size
            .nUBatch(512);         // physical (micro) batch size
        var context = new LlamaContext(arena, model, contextParams);

        // 5. Derive the vocab and tokenizer from the model.
        var vocab = new LlamaVocab(model);
        var tokenizer = new LlamaTokenizer(vocab, context);

        // 6. Tokenize some text. The response holds a native int buffer + token count.
        var tokens = tokenizer.tokenize(arena, "What is the capital of France?");
        System.out.println("Token count: " + tokens.size());

        // Read the first token id and decode it back to its UTF-8 piece.
        int firstToken = tokens.data().getAtIndex(ValueLayout.JAVA_INT, 0);
        byte[] piece = vocab.tokenToPiece(firstToken);
        System.out.println("First piece: " + new String(piece));

        // Inspect the model.
        System.out.println("Model: " + model.desc(arena));

        // 7. Free native resources in reverse creation order, then the Arena.
        context.free();
        model.free();
        LlamaRuntime.llama_backend_free();
        arena.close();
    }
}
```

## Options

### `LlamaModelParams`
| Method | Default | Purpose |
| --- | --- | --- |
| `nGpuLayers(int)` | model-dependent | Number of layers to offload to GPU (0 = CPU only). |
| `useMmap(boolean)` | `true` | Memory-map weights instead of copying into RAM. |
| `useMlock(boolean)` | `false` | Lock weights in RAM (prevents swap). |
| `vocabOnly(boolean)` | `false` | Load only the vocab (skips tensors; zeroes `nEmbdOut`/`desc`). |
| `splitMode(SplitMode)` | — | How to split weights across multiple GPUs. |
| `mainGpu(int)` | `0` | Primary GPU index. |

### `LlamaContextParams`
| Method | Default | Purpose |
| --- | --- | --- |
| `nCtx(int)` | from model | Context window size in tokens. |
| `nBatch(int)` | `512` | Logical batch size for a single `decode`. |
| `nUBatch(int)` | `512` | Physical (micro) batch size. |
| `nSeqMax(int)` | `1` | Max parallel sequences (see Parallel Conversations). |
| `nThreads(int)` / `nThreadsBatch(int)` | CPU count | Threads for generation / prompt processing. |
| `poolingType(PoolingType)` | model-dependent | Embedding pooling (see Embeddings / Reranking). |
| `embeddings(boolean)` | `false` | Output embeddings instead of logits. |

## Notes
- **Bootstrap order matters**: `LlamaLibLoader.load()` → `llama_backend_init()` → `ggml_backend_load_all_from_path(arena, libPath)` must run before any model is loaded. Override the lib directory with the `LLAMA_CPP_LIB_PATH` env var.
- **Context constructor takes the Arena**: use `new LlamaContext(arena, model, contextParams)` (3-arg). The model and context params are read at construction time.
- **`Freeable` resources**: `LlamaModel` and `LlamaContext` (and `LlamaSampler`) own native handles and must be `free()`d explicitly, in reverse order of creation (context before model). `free()` is idempotent-guarded — calling a method after `free()` throws.
- **Params vs. handles**: `LlamaModelParams`, `LlamaContextParams`, and `LlamaVocab` are backed by Arena-scoped segments (they extend `MemorySegmentAware`), so they are released when the `Arena` closes — you do not call `free()` on them.
- **One Arena to rule them**: allocate params and tokenizer buffers from a single confined `Arena` and `close()` it last; this frees all the small native allocations at once.
- **Construction can fail**: loading a missing/corrupt model or an invalid context configuration throws `LlamaException` (e.g. a quantized V-cache without flash attention).
- **Metal teardown caveat**: on Apple Silicon, every native handle must be freed before the JVM exits or llama.cpp can abort from a static destructor — always `free()` your model/context (in tests, track them for teardown).
- **`nGpuLayers(99)`** is the idiomatic "offload everything"; it is silently clamped to the model's layer count.

## See also
- [Text Generation & Sampling](../text-generation/README.md) — generate tokens with a sampler and iterator on top of the context built here.
- [Chat Templates](../chat-templates/README.md) — format multi-turn messages before tokenizing.
- [Devices & Memory](../device-and-memory/README.md) — enumerate backends/devices and estimate memory before loading.
- [Embeddings](../embeddings/README.md) — reuse this setup with `embeddings(true)` and a pooling type.
