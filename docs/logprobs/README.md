# Log Probabilities

> Collect per-token log-probabilities (the sampled token plus its top-N alternatives) for every token the model generates.

## Overview
Enabling logprobs makes each `LlamaOutput` carry a `Logprobs` object describing the model's confidence at that token position: the token actually sampled and the `topLogprobs` most-likely alternatives, sorted by descending log-probability. This mirrors the per-token logprobs returned by OpenAI-compatible chat-completion APIs when `logprobs=true`. Use it to inspect model confidence, build re-ranking or scoring logic, or surface alternative tokens in a UI. When disabled (the default), no extra logit processing is done.

## Key types
- `Logprobs` — record for one token position: `chosenToken()` and the `topLogprobs()` list.
- `TokenLogprob` — record for a single token: `token()`, `tokenId()`, `logprob()` (ln(p)), `bytes()` (raw UTF-8).
- `ConversationState.setTopLogprobs(int)` — opt-in switch; sets how many alternatives to collect (0 = off).
- `LlamaOutput.logprobs()` — returns the `Logprobs` for the emitted token, or `null` if collection is disabled.

## Usage
```java
import io.gravitee.llama.cpp.*;
import java.lang.foreign.Arena;
import java.nio.file.Path;

var arena = Arena.ofConfined();
LlamaRuntime.llama_backend_init();

var model = new LlamaModel(arena, Path.of("models/model.gguf"), new LlamaModelParams(arena));
var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512);
var context = new LlamaContext(arena, model, contextParams);
var vocab = new LlamaVocab(model);
var tokenizer = new LlamaTokenizer(vocab, context);
var sampler = new LlamaSampler(arena).temperature(0.7f).seed(42);

var state = ConversationState.create(arena, context, tokenizer, sampler)
    .setMaxTokens(50)
    .setTopLogprobs(5)   // return top-5 alternatives at every token position
    .initialize("What is the capital of France?");

var iterator = new DefaultLlamaIterator(state);
while (iterator.hasNext()) {
    var output = iterator.next();
    System.out.print(output.text());

    Logprobs lp = output.logprobs();          // null if setTopLogprobs was 0
    TokenLogprob chosen = lp.chosenToken();
    System.out.printf("%n  chosen: \"%s\"  logprob=%.4f%n", chosen.token(), chosen.logprob());
    lp.topLogprobs().forEach(t ->
        System.out.printf("    alt: \"%s\"  logprob=%.4f%n", t.token(), t.logprob()));
}

context.free();
sampler.free();
model.free();
LlamaRuntime.llama_backend_free();
```

## Options
| Method | Type | Default | Meaning |
| --- | --- | --- | --- |
| `setTopLogprobs(int)` | `int` | `0` | Number of top-alternative tokens to collect per position. `0` disables collection entirely (no overhead); max 20 by OpenAI convention. |
| `getTopLogprobs()` | `int` | — | Reads back the configured value. |

`Logprobs` / `TokenLogprob` fields:

| Field | Type | Notes |
| --- | --- | --- |
| `Logprobs.chosenToken()` | `TokenLogprob` | The token actually sampled at this position. |
| `Logprobs.topLogprobs()` | `List<TokenLogprob>` | Alternatives sorted by descending `logprob`; the chosen token is always included. |
| `TokenLogprob.token()` | `String` | Decoded token text (may be empty for special tokens). |
| `TokenLogprob.tokenId()` | `int` | Vocabulary ID of the token. |
| `TokenLogprob.logprob()` | `double` | `ln(p)`, always `<= 0`; `Double.NEGATIVE_INFINITY` if the model assigned probability zero. |
| `TokenLogprob.bytes()` | `List<Integer>` | Raw UTF-8 bytes of the token piece, useful for reassembling multi-byte characters split across tokens. |

## Notes
- Logprobs are opt-in: leave `setTopLogprobs` unset (or `0`) and `output.logprobs()` is `null` for every token, avoiding the cost of reading and sorting the full vocabulary logit vector.
- When enabled, `logprobs()` is non-null for every emitted token, and `chosenToken().token()` equals `output.content()` for that token.
- `topLogprobs()` contains at least `topN` entries; because the chosen token is always added, the list can be one longer if the chosen token was not already in the top-N.
- All `logprob` values are `<= 0` (they are natural logs of probabilities); the list is guaranteed sorted by descending `logprob`.
- Lifecycle: native handles (`LlamaModel`, `LlamaContext`, `LlamaSampler`) must be `free()`d (or allocated on a managed `Arena`); the `Logprobs`/`TokenLogprob` records are plain Java objects and need no cleanup.
- Sampler settings (temperature, seed, etc.) still drive which token is chosen; logprobs only report the per-position distribution and do not change sampling behavior.

## See also
- [Text Generation & Sampling](../text-generation/README.md) — configuring the sampler and iterating `LlamaOutput`.
- [Getting Started](../getting-started/README.md) — model/context/tokenizer setup used above.
- [Parallel Conversations (Batched Decoding)](../parallel-conversations/README.md) — running multiple `ConversationState`s at once.
- [Speculative Decoding](../speculative-decoding/README.md) — draft-model acceleration on the same `ConversationState`.
