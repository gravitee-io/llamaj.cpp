# Text Generation & Sampling

> Stream a model's response token-by-token from a single conversation, controlling the output with a configurable sampler chain, max-token and stop-string limits.

## Overview
This is the core single-conversation generation path: you wrap a context, tokenizer and sampler in a `ConversationState`, seed it with a prompt via `initialize(...)`, then drive a `DefaultLlamaIterator` to decode tokens one at a time. Each step yields a `LlamaOutput` (the decoded text piece plus token count and optional logprobs), and generation stops when the model emits an end-of-generation token, hits `setMaxTokens(...)`, or matches a configured stop string. The `LlamaSampler` chain decides *how* the next token is picked (greedy, temperature/top-k/top-p/min-p, mirostat, penalties, grammar, fixed seed).

## Key types
- `ConversationState` — holds the context/tokenizer/sampler plus prompt, sequence id and limits; created with `create(...)` and configured with fluent setters then `initialize(prompt)`.
- `DefaultLlamaIterator` — the autoregressive iterator over one state; exposes `stream()`, `hasNext()`/`next()`, and `close()` (it is `AutoCloseable`).
- `LlamaIterator<T>` — base class providing `stream()` over the iterator.
- `LlamaSampler` — a builder-style native sampler chain; each method (`temperature`, `topK`, `topP`, ...) appends a stage and returns `this`.
- `LlamaOutput` — record of one emitted step: `content()` / `text()`, `numberOfTokens()`, `sequenceId()`, `performance()`, `logprobs()`; `merge(other)` concatenates.
- `FinishReason` — why generation stopped: `EOS`, `STOP`, `LENGTH`, `TOOL_CALL`.

## Usage
```java
import io.gravitee.llama.cpp.*;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

var arena = Arena.ofConfined();
LlamaRuntime.llama_backend_init();

// Model + context
var model = new LlamaModel(arena, Path.of("models/model.gguf"), new LlamaModelParams(arena));
var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512);
var context = new LlamaContext(arena, model, contextParams);

// Tokenizer + a configured sampler chain (order matters: stages apply in sequence)
var vocab = new LlamaVocab(model);
var tokenizer = new LlamaTokenizer(vocab, context);
var sampler = new LlamaSampler(arena)
    .seed(new Random().nextInt())
    .temperature(0.75f)
    .topK(40)
    .topP(0.9f, 1)
    .minP(0.05f, 1)
    .penalties(64, 1.1f, 0.0f, 0.0f);

// Conversation state: prompt + generation limits
var state = ConversationState.create(arena, context, tokenizer, sampler)
    .setMaxTokens(100)
    .setStopStrings(List.of("\n\n", "User:"))
    .initialize("What is the capital of France?");

// Stream the response token-by-token
var iterator = new DefaultLlamaIterator(state);
String answer = iterator.stream()
    .reduce(LlamaOutput::merge)
    .orElse(new LlamaOutput("", 0))
    .content();
System.out.println(answer);

// Or consume incrementally:
// while (iterator.hasNext()) System.out.print(iterator.next().text());

System.out.println("finish reason: " + state.getFinishReason());      // EOS / STOP / LENGTH
System.out.println("in=" + state.getInputTokens() + " out=" + state.getAnswerTokens());

// Cleanup (native resources)
context.free();
sampler.free();
model.free();
LlamaRuntime.llama_backend_free();
```

## Options

### Sampler chain (`LlamaSampler`, fluent — each returns `this`)
| Method | Effect |
| --- | --- |
| `greedy()` | Always pick the argmax token (deterministic). |
| `temperature(float t)` | Scale logits; lower = sharper, higher = more random. |
| `topK(int k)` | Keep only the `k` most-likely tokens. |
| `topP(float p, int minKeep)` | Nucleus sampling: keep smallest set with cumulative prob ≥ `p` (at least `minKeep`). |
| `minP(float p, int minKeep)` | Drop tokens below `p` × top-token prob (at least `minKeep`). |
| `mirostat(int seed, float tau, float eta)` | Mirostat v2 adaptive-perplexity sampling. |
| `penalties(int lastN, float repeat, float freq, float present)` | Repetition / frequency / presence penalties over the last `lastN` tokens. |
| `grammar(LlamaVocab vocab, String grammar, String root)` | Constrain **all** output to a GBNF grammar. |
| `grammarLazy(LlamaVocab vocab, String grammar, String root, List<String> triggerPatterns, List<Integer> triggerTokens)` | Arm the grammar only once a trigger matches, and apply it from the pattern's **first capture group** onward. This is what makes a constrained region inside free text possible — constraining tool-call arguments with `grammar()` would force the model to answer in JSON even when it is only talking. Patterns are matched from the start of the generated output, so they normally read `^[\s\S]*?<escaped marker>([\s\S]*)`. |
| `seed(int seed)` | Append the final distribution sampler with a fixed RNG seed (reproducible). |

### Generation limits (`ConversationState`, fluent)
| Method | Effect |
| --- | --- |
| `setMaxTokens(int n)` | Cap generated answer tokens; `-1` (default) means unlimited (until EOG/context full). Triggers `FinishReason.LENGTH`. |
| `setStopStrings(List<String> stops)` | Stop as soon as the decoded tail matches any string; triggers `FinishReason.STOP`. |
| `setEogRamp(float startFraction, float maxBias)` | Budget-aware soft landing: bias end-of-generation as `maxTokens` nears, so the answer finishes a sentence instead of being severed. Off by default. See below. |
| `setTopLogprobs(int n)` | Attach top-`n` logprobs to each `LlamaOutput` (`0` = off). See the Log Probabilities doc. |
| `initialize(String prompt)` | Tokenize the prompt and (re)set all generation state — call last. |

### `FinishReason`
| Value | Meaning |
| --- | --- |
| `EOS` / `STOP` | Model emitted an end-of-generation token or matched a stop string. |
| `LENGTH` | `maxTokens` reached, or the context window filled. |
| `TOOL_CALL` | A tool-call section completed (see Reasoning & Tool Calls). |

## Budget-aware EOG ramp (soft landing)

`setMaxTokens(n)` alone is a guillotine: generation runs at full speed until the
counter trips, then stops mid-word. `setEogRamp(startFraction, maxBias)` turns the
budget into a pressure instead.

```java
state.setMaxTokens(250).setEogRamp(0.75f, 100f);
```

Past `startFraction * maxTokens`, every end-of-generation token's logit is raised
on a quadratic curve reaching `maxBias` at the cap. The boost is written into the
logits row immediately before sampling, so it behaves exactly like a `logit_bias`
sampler at the head of the chain and composes with temperature/top-k/top-p/penalties
downstream. Below the threshold nothing is written, so most of a run is bit-identical
to one without the ramp.

**The bias only applies where the text can stop.** After sentence-terminating
punctuation or a line break; from halfway up the ramp, also after clause punctuation
(`, ; : — )`); at the cap, anywhere. This gate is not an optimisation — without it the
ramp does not do what it looks like it does. Biasing every step makes EOG win wherever
it happens to overtake the next word, which is mid-clause, so the output is severed
exactly as before, only earlier. Gated, the bias can only take an exit the text had
already reached.

**`maxBias` sets how early it lands.** Mid-answer, EOG sits far below the running text
in raw logits — much further than probabilities suggest — so a small boost is simply
inert, and an inert ramp is indistinguishable from a disabled one. Measured on a
35B-A3B model at a 200-token budget: `12` never won a step (ran to the cap, severed),
`24` landed at 191, `100` landed at 169. Larger means more margin, not a different
ending; `100` is a sane default.

Effects to expect:

- Completions end **short** of the budget on a finished sentence — `max_tokens`
  becomes a target rather than a ceiling.
- The hard cap still applies as a backstop, for answers with no boundary anywhere in
  the ramp window (one long unbroken sentence, for instance).
- `FinishReason` is `LENGTH`, not `STOP`, when an EOG is produced while the ramp is
  active — the budget caused it, and agent loops rely on that to know the answer was
  cut short.
- Requires `maxTokens > 0`; with no budget the ramp is inert.
- Applies under speculative decoding too. Rejection sampling is exact for any target
  distribution provided the acceptance test and the residual draw see the same one,
  and both derive from the biased row. Verify rows are biased at their projected
  budget positions. The draft does not know about the bias, so acceptance dips for
  the last few tokens — a throughput cost, not a correctness one.

## Notes
- Call the fluent setters (`setMaxTokens`, `setStopStrings`, `setTopLogprobs`, ...) *before* `initialize(prompt)`; `initialize` resets generation state (and clears media) and tokenizes the prompt.
- Build the sampler chain in the order you want stages applied. The final stochastic pick comes from `seed(...)` (a distribution sampler); use `greedy()` instead for fully deterministic output.
- `LlamaSampler`, `LlamaContext` and `LlamaModel` own native memory — call `free()` on each (sampler and context are not freed by the iterator). In tests, `track(...)` your native resources so Metal buffers are released before JVM exit.
- `DefaultLlamaIterator` is `AutoCloseable`: `close()` removes the sequence from the KV cache (and frees speculative scratch). Use try-with-resources if you abandon a stream early; a fully consumed stream cleans up via `onFinished()`.
- `LlamaOutput.merge(...)` concatenates `content` and sums `numberOfTokens`, the idiomatic way to collect a full response from `stream()`.
- After the stream ends, read `state.getFinishReason()`, `state.getInputTokens()` and `state.getAnswerTokens()`; performance metrics are available via `iterator.getPerformance()` when the context is built with `noPerf(false)`.
- The prompt is decoded in chunks of `nBatch`; generation stops with `FinishReason.LENGTH` if the context window (`nCtx`) fills before EOG.

## See also
- [Getting Started](../getting-started/README.md) — minimal setup: backend init, model/context/sampler wiring.
- [Log Probabilities](../logprobs/README.md) — `setTopLogprobs(n)` and the `Logprobs` payload on each `LlamaOutput`.
- [Chat Templates](../chat-templates/README.md) — build the prompt string from system/user messages.
- [Parallel Conversations (Batched Decoding)](../parallel-conversations/README.md) — run many `ConversationState`s in one batch.
- [Speculative Decoding](../speculative-decoding/README.md) — draft/verify acceleration via `setDraft(...)` / `setNgram(...)`.
