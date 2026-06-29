# Speculative Decoding

> Speed up generation by having a cheap drafter propose several tokens that the target model verifies in a single pass, with byte-for-byte identical output.

## Overview
Speculative decoding accelerates token generation without changing the result: a cheap *drafter* proposes up to `nDraft` tokens per round and the target model verifies them all in one decode, committing the run of tokens it agrees with. The drafter is either a small **draft model** (`setDraft`) sharing the target's vocab, or **n-gram / prompt-lookup** matching against the generation history with no extra model (`setNgram`). Greedy configs are *lossless* (identical to plain greedy decoding); sampling configs use rejection sampling and stay an *exact* draw of the target distribution. Use it when you have spare compute and want lower latency — especially with repetitive output (code, JSON, RAG) for the n-gram path.

## Key types
- `SpeculativeConfig` — immutable record holding the draft window, sampling knobs, adaptive early-stop, and n-gram window; built via factories, withers, or a builder.
- `ConversationState` — per-conversation state; `setDraft(...)` / `setNgram(...)` enable speculation, `isSpeculative()` / `isNgram()` query it, `acceptRate()` reports accepted/drafted.
- `Speculation` (internal) — drives the draft→verify→accept cycle; the per-position distribution is computed natively, the accept test / residual draw / n-gram lookup are Java.
- `NgramIndex` (internal) — committed-token history with a position index for ~O(1) prompt-lookup proposals.

## Usage
```java
// Target + draft contexts over models that share the same tokenizer/vocab.
var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
var specCtx  = new LlamaContext(arena, model, cp);
var draftCtx = new LlamaContext(arena, model, cp); // a smaller model is the usual case
var vocab = new LlamaVocab(model);

var state = ConversationState.create(
        arena,
        specCtx,
        new LlamaTokenizer(vocab, specCtx),
        new LlamaSampler(arena).greedy())
    .setMaxTokens(24)
    .setDraft(draftCtx, SpeculativeConfig.greedy(4)) // greedy, fixed window of 4
    .initialize("The capital of France is");

String text = new DefaultLlamaIterator(state)
    .stream()
    .map(LlamaOutput::content)
    .reduce("", (a, b) -> a + b);

System.out.println(text + " (accept=" + state.acceptRate() + ")");
```

A `SpeculativeConfig` can be built three interchangeable ways:

```java
// 1. Factory presets:
SpeculativeConfig.greedy(4);                       // greedy, fixed window
SpeculativeConfig.greedyAdaptive(8, 1, 0.6f);      // greedy, stop early below draft confidence 0.6
new SpeculativeConfig(4, 0.8f, 40, 0.95f, 42);     // sampling (nDraft, temp, topK, topP, seed)

// 2. Fluent "withers" — start from a preset and override fields:
SpeculativeConfig.greedy(8).withTemperature(0.8f).withTopK(40).withTopP(0.95f).withSeed(42);

// 3. Builder:
SpeculativeConfig.builder().nDraft(8).temperature(0.8f).topK(40).topP(0.95f).seed(42).build();
```

N-gram (prompt-lookup) drafting needs no draft model — enable it with `setNgram`:

```java
state.setNgram(SpeculativeConfig.ngramGreedy(4, 2));                // greedy: kMax=4, ngram window=2
state.setNgram(SpeculativeConfig.ngram(4, 2, 0.8f, 40, 0.95f, 42)); // sampling
```

## From the CLI
`Main` (the bundled CLI) exposes speculation through flags — no code required:

| Flag | Meaning |
| --- | --- |
| `--draft <path>` | Draft model GGUF (model drafting); must share the target's vocab. Mutually exclusive with `--ngram`. |
| `--ngram <window>` | N-gram prompt-lookup drafting with this window; no draft model (default window 2). |
| `--n_draft <k>` | Max tokens drafted/proposed per round (default 4). |
| `--p_min <p>` | Adaptive early-stop: stop drafting once the draft's top-token probability drops below `p` (model drafting only; `0` = disabled). Distinct from `--min_p` (min-p sampling). |
| `--draft_min <n>` | Min tokens to draft before `--p_min` applies (default 1; clamped to `[1, n_draft]`). |

`--strategy DETERMINISTIC` selects the lossless greedy path; any temperature strategy (`CLASSIC_CHAT`/`FOCUSED`/`BALANCED`) makes speculation an exact memoryless sampler (temperature/top-k/top-p only). `CONSTRAINED`/`ADAPTIVE` and `--mmproj` are rejected up front (grammar/mirostat aren't memoryless; multimodal is unsupported).

```bash
# n-gram, lossless — great for repetitive output (code/JSON/RAG)
java -jar llamaj.cpp-<ver>.jar --model model.gguf --strategy DETERMINISTIC --ngram 2 --n_draft 4

# draft model, lossless, with adaptive early-stop
java -jar llamaj.cpp-<ver>.jar --model target.gguf --draft draft.gguf \
  --strategy DETERMINISTIC --n_draft 8 --p_min 0.6 --perf true
```

With `--perf true`, read the **`Effective generation`** line (emitted tokens ÷ wall-clock) and **`Speculative accept rate`** — the native `Generation speed` counter reads ~0 under speculation because batch-verified tokens aren't single-token decodes (so `n_eval` can't see them; they land under prompt eval). The realized speedup is **hardware-dependent**: largest on memory-bound datacenter GPUs where a verify batch is nearly as cheap as one decode, and more modest on Apple Silicon, where a multi-token verify batch costs meaningfully more than a single decode — so a high accept rate (more committed tokens per batch) is what makes it pay off there.

## Options
| Field | Factory / setter | Meaning |
| --- | --- | --- |
| `nDraft` | first ctor arg, `greedy(n)`, `.nDraft(n)`, `.withNDraft(n)` | Max tokens drafted/proposed per round (K / kMax), must be `>= 1`. |
| `temperature` | `.temperature(t)`, `.withTemperature(t)` | Sampling temperature; `0` selects the greedy (argmax) path. |
| `topK` | `.topK(k)`, `.withTopK(k)` | Top-K cutoff (`<= 0` disables). |
| `topP` | `.topP(p)`, `.withTopP(p)` | Top-P / nucleus cutoff (`>= 1` disables). |
| `seed` | `.seed(s)`, `.withSeed(s)` | RNG seed for draft sampling, the accept coin, and residual draws. |
| `draftMin` | `greedyAdaptive(max, min, pMin)`, `.draftMin(n)` | Min tokens to draft before `pMin` early-stop applies; clamped to `[1, nDraft]`. |
| `pMin` | `greedyAdaptive(...)`, `.pMin(p)`, `.withPMin(p)` | Draft-confidence floor; stop drafting once draft top-token prob drops below it (`<= 0` disables adaptive early stop). |
| `ngram` | `ngramGreedy(kMax, ngram)`, `ngram(...)`, `.ngram(n)`, `.withNgram(n)` | `0` = draft model (`setDraft`); `>= 1` = n-gram prompt-lookup (`setNgram`). |

Helpers: `isGreedy()` (`temperature <= 0`), `isAdaptive()` (`pMin > 0` and model-draft), `isNgram()` (`ngram > 0`). `builder()` defaults to greedy model-drafting with `nDraft == DEFAULT_N_DRAFT` (4); `toBuilder()` seeds a builder from an existing config.

## Notes
- **Only memoryless sampling is supported** inside speculation: `temperature`, `topK`, `topP`. Stateful/reshaping samplers (penalties, grammar, mirostat) are intentionally rejected because rejection sampling is exact only for memoryless distributions.
- **Native vs Java split:** the per-position distribution (temperature → top-k → top-p → softmax) is computed by llama.cpp's native sampler chain; the rejection-sampling accept test, residual draw, and n-gram lookup are Java (no native primitive exists for them).
- `setDraft` requires the draft context to share the target's vocab size (`draftContext.nVocab() == context.nVocab()`), otherwise it throws `LlamaException`. The conversation's main sampler is bypassed for accepted tokens — speculation is governed entirely by the `SpeculativeConfig`.
- `setNgram` requires an n-gram config (`config.isNgram()`, i.e. `ngram >= 1`); calling it with a model-draft config throws. A missing n-gram match simply degrades the round to a single target decode — never wrong output.
- **Lossless / exact regardless of draft quality:** the target verifies every proposed token and always commits at least one. A weaker draft only lowers `acceptRate()`, it never changes which tokens are emitted. `acceptRate()` is `0.0` until something is drafted.
- **Adaptive early stop** (`pMin > 0`) changes only *how many* tokens are speculated per round, not *which* are emitted, so it preserves greedy losslessness and sampling exactness.
- Works with both `DefaultLlamaIterator` (single sequence) and `BatchIterator` (fused multi-sequence). For the fused path, size the target context so `nBatch >= sum(nDraft + 1)` across sequences.
- **Resources/lifecycle:** the `Speculation` allocates persistent native scratch (sampler chain + draft/verify batches) on the state's `Arena`, freed once on teardown. Close the iterator (try-with-resources) when abandoning a stream so the scratch is freed and the sequence cleared; the contexts stay reusable afterward.

## See also
- [Text Generation & Sampling](../text-generation/README.md) — the underlying sampler chain and iterators speculation builds on.
- [Parallel Conversations (Batched Decoding)](../parallel-conversations/README.md) — the `BatchIterator` fused-step path used by speculative decoding.
- [Log Probabilities](../logprobs/README.md) — per-token logprobs, also driven from the candidate distribution.
- [Getting Started](../getting-started/README.md) — models, contexts, and arenas referenced above.
