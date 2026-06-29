# Quantized KV Cache

> Shrink the attention KV-cache footprint by storing the K and V tensors in a quantized `ggml_type` instead of the default F16.

## Overview
The KV cache grows linearly with context length and dominates memory use for long
contexts. By setting the K and V cache data types on `LlamaContextParams` to a quantized
format (for example `Q8_0` or `Q4_0`), you trade a small amount of accuracy for
substantially lower KV memory. The K cache can be quantized on its own, but a quantized V
cache requires flash attention to be enabled. Defaults are `F16` for both K and V.

## Key types
- `GgmlType` — enum of KV-suitable `ggml_type` values (`F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `IQ4_NL`, `BF16`); carries an explicit `nativeValue()`.
- `LlamaContextParams` — context configuration; `typeK(GgmlType)` / `typeV(GgmlType)` select the cache types, `flashAttnType(FlashAttentionType)` toggles flash attention.
- `FlashAttentionType` — `AUTO` / `DISABLED` / `ENABLED`; must be `ENABLED` for a quantized V cache.

## Usage
```java
import io.gravitee.llama.cpp.*;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;

public class QuantizedKvExample {
    public static void main(String[] args) {
        try (Arena arena = Arena.ofConfined()) {
            LlamaLibLoader.load();
            LlamaRuntime.llama_backend_init();

            var model = new LlamaModel(
                arena, Path.of("models/model.gguf"), new LlamaModelParams(arena));

            // Quantize the KV cache. A quantized V cache REQUIRES flash attention.
            var contextParams = new LlamaContextParams(arena)
                .nCtx(8192)
                .nBatch(512)
                .flashAttnType(FlashAttentionType.ENABLED)
                .typeK(GgmlType.Q8_0)
                .typeV(GgmlType.Q4_0);

            var context = new LlamaContext(model, contextParams);
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
}
```

You can also resolve a type from a CLI/string token: `GgmlType.fromString("q8_0")`.

## Options

### `GgmlType` values (KV-suitable)
| Constant | `nativeValue()` | `isQuantized()` | Notes |
|----------|-----------------|-----------------|-------|
| `F32`    | 0  | false | Full precision, largest |
| `F16`    | 1  | false | Default K/V type |
| `Q4_0`   | 2  | true  | Smallest common quant |
| `Q4_1`   | 3  | true  | |
| `Q5_0`   | 6  | true  | Native value gaps (legacy quants removed upstream) |
| `Q5_1`   | 7  | true  | |
| `Q8_0`   | 8  | true  | Good accuracy/size trade-off |
| `IQ4_NL` | 20 | true  | Non-linear 4-bit |
| `BF16`   | 30 | false | Brain-float, full-precision-class |

### `FlashAttentionType`
| Value | Effect |
|-------|--------|
| `AUTO` | Let llama.cpp decide (default) |
| `DISABLED` | Force off |
| `ENABLED` | Force on — required for a quantized V cache |

### `LlamaContextParams` KV setters
| Method | Default | Purpose |
|--------|---------|---------|
| `typeK(GgmlType)` | `F16` | K cache data type |
| `typeV(GgmlType)` | `F16` | V cache data type (quantized ⇒ needs flash attention) |
| `flashAttnType(FlashAttentionType)` | `AUTO` | Enable flash attention |

## Notes
- A quantized **V** cache requires flash attention: set `flashAttnType(FlashAttentionType.ENABLED)` or constructing the `LlamaContext` will fail. Quantizing only **K** does not require it.
- The native `ggml_type` enum is non-contiguous (gaps where legacy quants like `Q4_2`/`Q4_3` were removed). Always map via `GgmlType.nativeValue()` / `GgmlType.fromNative(int)`; never use `ordinal()`.
- The K-quant family (`Q4_K`, `Q5_K`, …) is intentionally **not** exposed: its 256-element block size does not divide typical attention head dimensions, so it is invalid for the KV cache.
- `GgmlType.fromString(...)` and `fromNative(...)` throw `LlamaException` for unknown/unsupported tokens.
- `LlamaContextParams` is allocated from an `Arena`; the `LlamaContext`, model, and sampler are native resources — call `free()` on each (and `llama_backend_free()`) when done. In tests, track native resources so Metal buffers are released before JVM exit.
- Setters are fluent (return `LlamaContextParams`), so they chain with other context options.

## See also
- [Getting Started](../getting-started/README.md) — basic model/context setup the example builds on.
- [Devices & Memory](../device-and-memory/README.md) — complementary ways to manage memory and offloading.
- [Text Generation & Sampling](../text-generation/README.md) — the iterator/sampler loop used above.
- [Speculative Decoding](../speculative-decoding/README.md) — cut latency with a draft model or n-gram lookup.
