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
| `grammar(LlamaVocab vocab, String grammar, String root)` | Constrain output to a GBNF grammar. |
| `seed(int seed)` | Append the final distribution sampler with a fixed RNG seed (reproducible). |

### Generation limits (`ConversationState`, fluent)
| Method | Effect |
| --- | --- |
| `setMaxTokens(int n)` | Cap generated answer tokens; `-1` (default) means unlimited (until EOG/context full). Triggers `FinishReason.LENGTH`. |
| `setStopStrings(List<String> stops)` | Stop as soon as the decoded tail matches any string; triggers `FinishReason.STOP`. |
| `setTopLogprobs(int n)` | Attach top-`n` logprobs to each `LlamaOutput` (`0` = off). See the Log Probabilities doc. |
| `initialize(String prompt)` | Tokenize the prompt and (re)set all generation state — call last. |

### `FinishReason`
| Value | Meaning |
| --- | --- |
| `EOS` / `STOP` | Model emitted an end-of-generation token or matched a stop string. |
| `LENGTH` | `maxTokens` reached, or the context window filled. |
| `TOOL_CALL` | A tool-call section completed (see Reasoning & Tool Calls). |

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
