# Parallel Conversations (Batched Decoding)

> Decode many independent conversations together in a single shared context, each isolated by a distinct sequence id.

## Overview
`BatchIterator` runs several conversations through **one** `LlamaContext` at the same time, packing one token per active conversation into each decode step. Every conversation is a `ConversationState` tagged with a unique **sequence id**, which keeps each one's KV cache logically separate inside the shared context. Use this when you need to serve multiple prompts concurrently (e.g. a chat server handling several clients) and want to amortise the cost of each forward pass across all of them.

## Key types
- `BatchIterator` — the parallel iterator; holds the shared context, manages per-sequence state, and yields one `LlamaOutput` per token via `hasNext()`/`next()` or `stream()`.
- `ConversationState` — one conversation: its prompt, sampler, tokenizer, generation/finish state, and its `sequenceId`.
- `LlamaContext` — the single context shared by all states; its `nSeqMax()` caps how many sequences can run in parallel.
- `LlamaOutput` — a record per emitted token, carrying `text()`/`content()`, `sequenceId()`, and optional `logprobs()`.

## Usage
```java
import io.gravitee.llama.cpp.*;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

var arena = Arena.ofConfined();
LlamaRuntime.llama_backend_init();

var model = new LlamaModel(arena, Path.of("models/model.gguf"), new LlamaModelParams(arena));

// One context, sized to host several sequences at once.
var contextParams = new LlamaContextParams(arena)
    .nCtx(2048)
    .nBatch(512)
    .nSeqMax(4);            // up to 4 parallel sequences
var context = new LlamaContext(arena, model, contextParams);

var vocab     = new LlamaVocab(model);
var tokenizer = new LlamaTokenizer(vocab, context);
var sampler   = new LlamaSampler(arena).seed(new Random().nextInt());

// Each conversation gets a UNIQUE sequence id (0, 1, 2) and shares the same context.
var state1 = ConversationState.create(arena, context, tokenizer, sampler, 0)
    .setMaxTokens(50).initialize("What is the capital of France?");
var state2 = ConversationState.create(arena, context, tokenizer, sampler, 1)
    .setMaxTokens(50).initialize("What is the capital of England?");
var state3 = ConversationState.create(arena, context, tokenizer, sampler, 2)
    .setMaxTokens(50).initialize("What is the capital of Poland?");

// Prompts are auto-processed when states are added.
var parallel = new BatchIterator(arena, context)
    .addState(state1)
    .addState(state2)
    .addState(state3);

// Accumulate output per sequence — each token names which conversation it belongs to.
Map<Integer, StringBuilder> answers = new HashMap<>();
answers.put(0, new StringBuilder());
answers.put(1, new StringBuilder());
answers.put(2, new StringBuilder());

parallel.stream()
    .forEach(out -> answers.get(out.sequenceId()).append(out.text()));

// Equivalent explicit loop:
// while (parallel.hasNext()) {
//   LlamaOutput out = parallel.next();
//   answers.get(out.sequenceId()).append(out.text());
// }

parallel.free();
context.free();
sampler.free();
model.free();
LlamaRuntime.llama_backend_free();
```

## Options
| Knob | Where | Role |
| --- | --- | --- |
| `nSeqMax(int)` | `LlamaContextParams` | Maximum number of distinct sequence ids that can run in parallel; must be ≥ the number of states you add. |
| `nBatch(int)` | `LlamaContextParams` | Tokens decoded per step; the iterator decodes up to `nBatch` active sequences per batch (one token each). |
| `nCtx(int)` | `LlamaContextParams` | Total KV cache budget shared across all sequences. |
| `sequenceId` | `ConversationState.create(...)` | Unique id distinguishing this conversation in the shared context (the 4-arg `create` defaults it to `0`). |
| `setMaxTokens(int)` | `ConversationState` | Per-conversation generation cap. |

## Notes
- **Distinct sequence ids are mandatory.** Adding two states with the same id throws `LlamaException` ("Sequence ID N is already in use").
- **All states must share the iterator's context.** Adding a state built on a different `LlamaContext` throws `LlamaException` ("All conversation states must share the same LlamaContext").
- **`nSeqMax` is the ceiling.** Size it for the most concurrent conversations you expect; the underlying batch is allocated with `context.nSeqMax()`.
- **`addState` is incremental and chainable**, and is thread-safe — you can add a conversation while the iterator is already running. Prompts are processed lazily on the next batch iteration, and each conversation's first token is emitted as soon as its prompt is prefilled.
- **`removeState(int sequenceId)`** cancels one conversation (e.g. a disconnected client) and frees its KV cache; returns `true` if found. **`stop()`** halts everything and clears all sequences; after it, `hasNext()` and `hasActiveConversations()` return `false`.
- **Lifecycle.** `BatchIterator` is `AutoCloseable`: call `free()` (or use try-with-resources) to stop and release the batch; it does **not** free the shared `context`, `sampler`, or `model`, which you free separately. `free()`/`stop()` are idempotent.
- **`next()` requires `hasNext()` first** — calling `next()` with nothing queued throws `NoSuchElementException`.
- Each conversation tracks its own `getFinishReason()`, `getAnswerTokens()`, and other counters independently.

## See also
- [Getting Started](../getting-started/README.md) — model/context/sampler setup used here.
- [Text Generation & Sampling](../text-generation/README.md) — single-conversation generation and sampler configuration.
- [Speculative Decoding](../speculative-decoding/README.md) — draft/verify acceleration, also batched per sequence by `BatchIterator`.
- [Log Probabilities](../logprobs/README.md) — the `logprobs()` carried on each `LlamaOutput`.
