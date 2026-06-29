# Chat Templates

> Render a list of role-tagged chat messages into a single prompt string using the model's built-in Jinja2 chat template.

## Overview
Instruction-tuned GGUF models ship with a chat template (e.g. Qwen3's `<|im_start|>` / `<|im_end|>` markers) that defines how `system`, `user`, and `assistant` turns are formatted. `LlamaTemplate` reads that template from the model and applies it to a `LlamaChatMessages` list, returning the formatted prompt string you then tokenize and feed to a `LlamaContext`. Use it whenever you generate from a chat/instruct model so the prompt matches what the model was trained on.

## Key types
- `LlamaTemplate` — wraps the model's chat template; `applyTemplate(...)` formats messages, `templateString()` returns the raw Jinja2 template.
- `LlamaChatMessage` — one message, built from an `Arena`/allocator, a `Role`, and a `String` content.
- `LlamaChatMessages` — an ordered, contiguous native array of `LlamaChatMessage` passed to `applyTemplate`.
- `Role` — enum of the supported roles: `SYSTEM`, `USER`, `ASSISTANT`.

## Usage
```java
try (Arena arena = Arena.ofConfined()) {
  var model = new LlamaModel(arena, Path.of("models/model.gguf"), new LlamaModelParams(arena));
  var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512);

  // Build the conversation
  var messages = new LlamaChatMessages(
    arena,
    List.of(
      new LlamaChatMessage(arena, Role.SYSTEM, "You are a helpful assistant"),
      new LlamaChatMessage(arena, Role.USER, "What's the capital of France?")
    )
  );

  // Apply the model's chat template -> formatted prompt string
  var template = new LlamaTemplate(model);
  String prompt = template.applyTemplate(arena, messages, contextParams.nCtx());

  // (optional) inspect the raw Jinja2 template the model ships with
  String raw = template.templateString(); // null if the model has no template

  System.out.println(prompt); // feed `prompt` to tokenization / generation
}
```

## Options
| Knob | Where | Meaning |
| --- | --- | --- |
| `Role.SYSTEM` / `Role.USER` / `Role.ASSISTANT` | `new LlamaChatMessage(arena, role, content)` | The role tag rendered by the template; mapped to native labels `"system"`, `"user"`, `"assistant"`. |
| `nCtx` | `applyTemplate(arena, messages, nCtx)` | Initial size of the output buffer; the call auto-grows and retries if the rendered prompt is longer. |

## Notes
- `applyTemplate` always renders with the add-generation-prompt flag enabled, so the output ends with the assistant turn opener ready for generation.
- The buffer starts at `nCtx` chars; if the template needs more, `LlamaTemplate` reallocates to the required length and renders again, so passing your context size is safe.
- A negative result from the native call raises `IllegalStateException("failed to apply the chat template.")`.
- `templateString()` returns `null` when the model carries no embedded chat template; otherwise it is the raw Jinja2 source (useful for debugging which markers a model expects).
- All three types extend `MemorySegmentAware` and allocate from the supplied `Arena`/`SegmentAllocator`; scope them in a try-with-resources `Arena.ofConfined()` so native memory is freed when the arena closes. `LlamaChatMessages` copies each message struct into one contiguous native array, so keep the messages alive for the duration of the call.
- `Role.fromLabel(label)` throws `IllegalArgumentException("Invalid role: ...")` for unknown labels; only the three roles above are supported.

## See also
- [Getting Started](../getting-started/README.md) — load a model and create a context before templating.
- [Text Generation & Sampling](../text-generation/README.md) — tokenize the templated prompt and generate a response.
- [Reasoning & Tool Calls](../reasoning-and-tool-calls/README.md) — structured chat with reasoning traces and tool invocations.
- [Multimodal (Vision & Audio)](../multimodal/README.md) — chat messages combined with image/audio inputs.
