# Reasoning & Tool Calls

> Tag each generated token as reasoning, answer, or tool-call output by giving the conversation the delimiter strings that wrap each section.

## Overview
Many models interleave their output: a chain-of-thought block (e.g. `<think>…</think>`), a tool-call block (e.g. `<tool_call>…</tool_call>`), and the final user-facing answer. By registering the start/end delimiters for each section on a `ConversationState`, the iterator detects the active `GenerationState` while streaming, lets you count tokens per section separately, and surfaces `FinishReason.TOOL_CALL` when generation ends inside a tool call. Use this when you need to hide/strip reasoning, route tool-call payloads to a tool executor, or bill reasoning vs answer tokens differently.

## Key types
- `GenerationState` — enum of the section the model is currently emitting: `ANSWER`, `REASONING`, `TOOLS`.
- `StateBounds` — record `(GenerationState state, String start, String end)` pairing a section with its delimiter strings.
- `ConversationState` — owns the generation; `setReasoning(start, end)` / `setToolCall(start, end)` register the delimiters before `initialize(prompt)`.
- `FinishReason` — terminal reason; `TOOL_CALL` (label `"tool_calls"`) is reported when output ends inside a tool-call block, alongside `EOS` / `STOP` / `LENGTH`.
- `DefaultLlamaIterator` — drives token-by-token generation; `getGenerationState()` on the state reflects the active section as you stream.

## Usage
```java
var arena = Arena.ofConfined();

var model = new LlamaModel(arena, modelPath, new LlamaModelParams(arena));
var contextParams = new LlamaContextParams(arena);
var context = new LlamaContext(arena, model, contextParams);
var vocab = new LlamaVocab(model);
var tokenizer = new LlamaTokenizer(vocab, context);
var sampler = new LlamaSampler(arena).seed(42);

var prompt = getPrompt(model, arena, buildMessages(arena, system, input), contextParams);

// Register the delimiters BEFORE initialize(...): reasoning + tool-call detection.
var state = ConversationState.create(arena, context, tokenizer, sampler)
  .setReasoning("<think>", "</think>")
  .setToolCall("<tool_call>", "</tool_call>")
  .initialize(prompt);

var it = new DefaultLlamaIterator(state);

// Merge all streamed pieces into a single output (content + token count).
LlamaOutput output = it.stream()
  .reduce(LlamaOutput::merge)
  .orElse(new LlamaOutput("", 0));

// Per-section token counts.
int inputTokens     = state.getInputTokens();
int answerTokens    = state.getAnswerTokens();
int reasoningTokens = state.getReasoningTokens();
int toolCallTokens  = state.getToolsTokens();

// output.numberOfTokens() == answerTokens + reasoningTokens + toolCallTokens
// state.getTotalTokenCount() == inputTokens + answerTokens + reasoningTokens + toolCallTokens

if (state.getFinishReason() == FinishReason.TOOL_CALL) {
  // Output ended inside <tool_call>...</tool_call> — dispatch to your tool executor.
}

context.free();
sampler.free();
model.free();
```

To act on sections as they stream (rather than after the fact), inspect `state.getGenerationState()` inside the stream — it returns the active `GenerationState` for the token just produced:

```java
it.stream().forEach(chunk -> {
  switch (state.getGenerationState()) {
    case REASONING -> { /* hide or log chain-of-thought */ }
    case TOOLS     -> { /* accumulate tool-call JSON */ }
    case ANSWER    -> { /* forward to the user */ }
  }
});
```

## Options
| Method | Section | Effect |
| --- | --- | --- |
| `setReasoning(String start, String end)` | `REASONING` | Registers a `StateBounds(REASONING, start, end)`; tokens between the delimiters count toward `getReasoningTokens()`. |
| `setToolCall(String start, String end)` | `TOOLS` | Registers a `StateBounds(TOOLS, start, end)`; tokens between the delimiters count toward `getToolsTokens()`; ending here yields `FinishReason.TOOL_CALL`. |
| `getReasoningTokens()` / `getToolsTokens()` / `getAnswerTokens()` | — | Per-section output token counts; `getInputTokens()` and `getTotalTokenCount()` cover the prompt and grand total. |

| `GenerationState` | Meaning |
| --- | --- |
| `ANSWER` | Default state; the user-facing answer (everything outside a registered section). |
| `REASONING` | Inside the reasoning delimiters configured via `setReasoning`. |
| `TOOLS` | Inside the tool-call delimiters configured via `setToolCall`. |

## Notes
- Configure `setReasoning` / `setToolCall` **before** `initialize(prompt)` — `initialize` resets generation state (and re-reads the registered `StateBounds`). Both methods are chainable and return the same `ConversationState`.
- The delimiter strings must match what the model actually emits (driven by its chat template / system prompt). The tool-call test, for example, instructs the model via the system prompt to wrap calls in `<tool_call>…</tool_call>` and then registers those exact delimiters.
- These delimiters are not appended to the prompt; they are pure output classifiers. Detection works on the decoded text stream, so multi-token delimiters are handled.
- `getAnswerTokens()` always reflects the `ANSWER` section even when reasoning/tool calls are configured; the three section counts plus the input count sum to `getTotalTokenCount()`.
- `FinishReason.TOOL_CALL` is a marker that generation stopped inside a tool call; the natural terminal reasons remain `EOS`, `STOP`, and `LENGTH`. `isFinished()` distinguishes a real stop (EOG/length) from a marker set while generation could otherwise continue.
- Resource lifecycle: the model, context, and sampler are native resources — `free()` them (or close the owning `Arena`) when done, in the reverse of allocation, to release Metal/CPU buffers before JVM exit.

## See also
- [Text Generation & Sampling](../text-generation/README.md) — the base streaming/iterator loop these section detectors build on.
- [Chat Templates](../chat-templates/README.md) — building the prompt that makes models emit reasoning and tool-call blocks.
- [Log Probabilities](../logprobs/README.md) — another per-token signal collected alongside generation via `setTopLogprobs`.
- [Parallel Conversations (Batched Decoding)](../parallel-conversations/README.md) — running many `ConversationState`s, each with its own section tracking, in one batch.
