# Reranking

> Score how relevant documents are to a query using a cross-encoder, then sort the candidates by score.

## Overview
`LlamaReranker` is a high-level wrapper over `LlamaContext` that produces a relevance
score for a `(query, document)` pair. Use it to re-order a candidate set retrieved by a
first-pass search (e.g. an embedding/vector search): score each document against the
query, then sort descending. Pooling (`RANK`) and attention are auto-detected from the
GGUF architecture, so `Options.defaults()` works for most reranker models, and the wrapper
supports both BERT-family cross-encoders (single raw logit) and chat-style rerankers like
Qwen3-Reranker (two-class softmax) through a pluggable `RerankTemplate`.

## Key types
- `LlamaReranker` — cross-encoder wrapper; `score(query, doc)` and `scoreAll(query, docs)` return raw `float[]` scores. Implements `AutoCloseable`/`Freeable`.
- `LlamaReranker.Options` — record of `nCtx`, `nBatch`, `nSeqMax`, `attention`, `template`; build with `Options.defaults()` plus `withXxx(...)`.
- `RerankTemplate` — functional interface `(query, document) -> String` that formats the tokenizer input. `RerankTemplate.PLAIN` is the default (`query + " " + document`).
- `LlamaModel` — the loaded GGUF; the caller owns and frees it (the reranker does not).

## Usage

### Cross-encoder (BERT-family, e.g. Jina / BGE reranker)

```java
import io.gravitee.llama.cpp.*;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

public class RerankExample {
    public static void main(String[] args) {
        var arena = Arena.ofConfined();
        String libPath = LlamaLibLoader.load();
        LlamaRuntime.llama_backend_init();
        LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

        var model = new LlamaModel(arena, Path.of("models/reranker.gguf"), new LlamaModelParams(arena));
        var reranker = new LlamaReranker(arena, model, LlamaReranker.Options.defaults());

        String query = "What is the capital of France?";
        List<String> documents = List.of(
            "Paris is the capital and most populous city of France.",
            "The Eiffel Tower is a landmark in Paris.",
            "Bananas are a good source of potassium."
        );

        // One raw score array per document, in input order (batched decode under the hood).
        List<float[]> scores = reranker.scoreAll(query, documents);

        // BERT cross-encoder: scores[i][0] is a single raw logit, higher = more relevant.
        IntStream.range(0, documents.size())
            .boxed()
            .sorted(Comparator.comparingDouble((Integer i) -> scores.get(i)[0]).reversed())
            .forEach(i -> System.out.printf("%.4f  %s%n", scores.get(i)[0], documents.get(i)));

        reranker.close(); // frees only the context; the model is owned by the caller
        model.free();
        LlamaRuntime.llama_backend_free();
        arena.close();
    }
}
```

For a single pair, `reranker.score(query, document)` returns a `float[]` of size
`reranker.nClsOut()` — `1` for a BERT cross-encoder.

### Chat-style reranker (Qwen3-Reranker) with a custom `RerankTemplate`

Qwen3-Reranker expects the pair wrapped in a system/user prompt and has a 2-class head:
`nClsOut() == 2` and `score` returns a softmax `float[2]` where index `0` is `P(relevant)`.

```java
// Wraps the pair in Qwen3's chat format, instructing the model to answer yes/no.
static final RerankTemplate QWEN3_TEMPLATE = (query, document) ->
    "<|im_start|>system\n" +
    "Judge whether the Document is relevant to the Query. Answer 'yes' or 'no'." +
    "<|im_end|>\n" +
    "<|im_start|>user\n" +
    "Query: " + query + "\n" +
    "Document: " + document + "\n" +
    "Relevant:<|im_end|>\n";

var reranker = new LlamaReranker(
    arena,
    model,
    LlamaReranker.Options.defaults().withTemplate(QWEN3_TEMPLATE)
);

List<float[]> scores = reranker.scoreAll(query, documents);
// Rank by P(relevant) = scores.get(i)[0], highest first.
```

## Options

| Field / setter | Default (when `null`) | Purpose |
| --- | --- | --- |
| `nCtx` / `withNCtx` | `0` → model's trained context (`n_ctx_train`) | Context size in tokens. |
| `nBatch` / `withNBatch` | llama.cpp default (e.g. `2048`) | Max tokens packed per decode; also the per-pair token cap. |
| `nSeqMax` / `withNSeqMax` | llama.cpp default (e.g. `1`) | Max sequences packed per decode; higher = more parallelism, more KV memory. |
| `attention` / `withAttention` | auto: `NON_CAUSAL` (encoder) / `CAUSAL` (decoder) | `AttentionType` for the context. |
| `template` / `withTemplate` | `RerankTemplate.PLAIN` | `(query, document) -> String` input formatter. |

Output interpretation by model family:

| Model family | `nClsOut()` | `score(...)` meaning |
| --- | --- | --- |
| BERT cross-encoder (BGE, Jina) | `1` | A single unbounded relevance **logit**; apply `sigmoid` only if you need a `[0,1]` value. |
| Qwen3-Reranker | `2` | Softmax `[P(yes), P(no)]`; `score[0]` is the `[0,1]` relevance probability. |

## Notes
- Ranking is identical for both families: sort by `score[i][0]` descending. Only the
  absolute interpretation differs (raw logit vs. calibrated probability).
- The GGUF must support `RANK` pooling (a classifier head must be present); otherwise
  `getEmbeddingsSeq` returns `null` and `scoreAll` throws a `LlamaException`.
- `scoreAll` packs multiple pairs into one `LlamaBatch` when they fit under `nBatch`
  tokens / `nSeqMax` sequences and decodes them together — faster than one decode per
  document. Batched vs. single-call scores are numerically equivalent but not bit-exact.
- Any single formatted pair that tokenises to more than `nBatch` tokens throws a
  `LlamaException`; raise `Options.nBatch` or truncate the document.
- The caller owns the `LlamaModel`. `free()`/`close()` releases only the internally-created
  `LlamaContext`; you must still `model.free()` and close your `Arena`.
- `LlamaReranker` is **not thread-safe**: use one instance per thread or synchronise externally.

## See also
- [Embeddings](../embeddings/README.md) — produce the dense vectors for the first-pass retrieval that reranking re-orders.
- [Chat Templates](../chat-templates/README.md) — how prompt formatting works for chat-style models.
- [Parallel Conversations (Batched Decoding)](../parallel-conversations/README.md) — the batched-decode mechanism `scoreAll` builds on.
- [Getting Started](../getting-started/README.md) — backend init, model loading, and Arena lifecycle.
