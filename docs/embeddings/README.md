# Embeddings

> Turn text into dense float vectors with `LlamaEmbedder`, a high-level wrapper that handles pooling, attention, tokenization and batching for you.

## Overview
`LlamaEmbedder` produces a dense embedding vector (`float[]`) for a piece of text, suitable for semantic search, clustering, RAG retrieval and cosine-similarity comparisons. It wraps a `LlamaContext` created in embedding mode and reduces the API to `String -> float[]`. Pooling and attention are auto-detected from the GGUF architecture, so `Options.defaults()` works for most embedding models (BERT-family encoders as well as decoder embedding models like Qwen3-Embedding). Vectors are returned **un-normalised** — L2-normalise them yourself before computing cosine similarity.

## Key types
- `LlamaEmbedder` — high-level wrapper; `embed(String)` for one text, `embedAll(List<String>)` to batch many through a single decode. Implements `Freeable`/`AutoCloseable`.
- `LlamaEmbedder.Options` — record of `nCtx`, `nBatch`, `nSeqMax`, `pooling`, `attention`; any `null` field is auto-detected. Use `Options.defaults()` plus `withX(...)` overrides.
- `PoolingType` — how token states are pooled into one vector: `CLS`, `MEAN`, `LAST`, `RANK`, `NONE`, `UNSPECIFIED`.
- `AttentionType` — `CAUSAL` (decoders) or `NON_CAUSAL` (encoders); `UNSPECIFIED` for auto.
- `LlamaModel` — caller-owned, pre-loaded model; reused across embedders and **not** freed by the embedder.

## Usage
```java
import io.gravitee.llama.cpp.*;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;

public class EmbeddingExample {
    static float[] l2normalize(float[] v) {
        double sum = 0;
        for (float f : v) sum += (double) f * f;
        double norm = Math.sqrt(sum);
        if (norm > 1e-9) for (int i = 0; i < v.length; i++) v[i] = (float) (v[i] / norm);
        return v;
    }

    static double cosine(float[] a, float[] b) {
        double dot = 0;
        for (int i = 0; i < a.length; i++) dot += (double) a[i] * b[i];
        return dot; // inputs are L2-normalised, so the dot product is the cosine
    }

    public static void main(String[] args) {
        var arena = Arena.ofConfined();
        String libPath = LlamaLibLoader.load();
        LlamaRuntime.llama_backend_init();
        LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

        var model = new LlamaModel(arena, Path.of("models/embedding.gguf"), new LlamaModelParams(arena));
        var embedder = new LlamaEmbedder(arena, model, LlamaEmbedder.Options.defaults());

        System.out.println("embedding dimension: " + embedder.nEmbdOut());
        System.out.println("pooling: " + embedder.poolingType());

        // Single text
        float[] one = embedder.embed("The capital of France is Paris.");

        // Or batch many through a single decode (same order out as in)
        List<float[]> embeddings = embedder.embedAll(List.of(
            "The capital of France is Paris.",
            "Paris is France's largest city.",
            "Bananas are a good source of potassium."
        ));
        embeddings.forEach(EmbeddingExample::l2normalize);

        System.out.printf("similar pair:   %.4f%n", cosine(embeddings.get(0), embeddings.get(1)));
        System.out.printf("unrelated pair: %.4f%n", cosine(embeddings.get(0), embeddings.get(2)));

        embedder.close(); // frees only the internal context
        model.free();     // caller owns the model
        LlamaRuntime.llama_backend_free();
        arena.close();
    }
}
```

## Options
`LlamaEmbedder.Options` — every field is nullable; `null` means auto-detect / use llama.cpp's default.

| Field | Type | Default (`null`) | Notes |
|-------|------|------------------|-------|
| `nCtx` | `Integer` | `0` → model's trained context (`n_ctx_train`) | Context size in tokens. |
| `nBatch` | `Integer` | llama.cpp default (`2048`) | Max tokens per decode; a single input exceeding this throws `LlamaException`. |
| `nSeqMax` | `Integer` | llama.cpp default (`1`) | Sequences packed per decode in `embedAll`; higher = more parallelism, more KV memory. |
| `pooling` | `PoolingType` | `CLS` for encoders, `LAST` for decoders | How token states collapse into one vector. |
| `attention` | `AttentionType` | `NON_CAUSAL` for encoders, `CAUSAL` for decoders | Encoders attend bidirectionally; decoders are left-to-right. |

Build with `Options.defaults()` and chain overrides, e.g. `Options.defaults().withPooling(PoolingType.MEAN).withNSeqMax(8)`.

**`PoolingType` values:** `UNSPECIFIED`, `NONE`, `MEAN`, `CLS`, `LAST`, `RANK`.
**`AttentionType` values:** `UNSPECIFIED`, `CAUSAL`, `NON_CAUSAL`.

## Notes
- The embedder always builds its `LlamaContext` with `embeddings(true)`; you do not set this yourself. If you drop down to the raw `LlamaContext` API, the context must be created in embedding mode or `getEmbeddingsSeq`/`getEmbeddingsIth` return null.
- Auto-detection reads the GGUF `general.architecture` metadata: BERT-family encoders (`bert`, `nomic-bert`, `modern-bert`, `jina-bert-*`, `neo-bert`, `eurobert`) get `CLS` + `NON_CAUSAL`; all other (decoder) architectures get `LAST` + `CAUSAL`.
- Vectors are **un-normalised** — apply L2 normalisation before cosine similarity (cosine then reduces to a dot product).
- `embed`/`embedAll` return fresh `float[]` copies of length `nEmbdOut()`; the same text yields a deterministic result.
- `embedAll` packs sequences until they fill `nBatch` tokens or `nSeqMax` sequences, then decodes the batch once — much faster than one decode per text. Batched and single-call outputs are semantically equivalent (cosine ≈ 1.0) but not bit-exact, since packing changes the FP compute order.
- Resource ownership: the caller owns the `LlamaModel` and must `free()` it. `close()`/`free()` on the embedder releases only its internal context, so one model can back multiple embedder / reranker / classifier instances.
- Not thread-safe — create one embedder per thread or synchronise externally.
- Lifecycle: pass a confined `Arena`, and at shutdown close the embedder, free the model, call `LlamaRuntime.llama_backend_free()`, then close the arena (track native resources so Metal buffers free before JVM exit).

## See also
- [Reranking](../reranking/README.md) — score query/document relevance using the same model-as-context pattern (`RANK` pooling).
- [Getting Started](../getting-started/README.md) — backend init, model loading and arena lifecycle.
- [Text Generation & Sampling](../text-generation/README.md) — generative inference on a non-embedding context.
- [Quantized KV Cache](../quantized-kv-cache/README.md) — reduce KV memory when packing many sequences.
