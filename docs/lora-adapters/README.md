# LoRA Adapters

> Load a GGUF LoRA adapter and attach it to a model so generation runs with the fine-tuned weights.

## Overview
A LoRA (Low-Rank Adaptation) adapter is a small set of fine-tuned weights, packaged as its own GGUF file, that specializes a base model for a task (e.g. summarization, a particular style) without re-loading the full model. In llamaj.cpp you load the base model normally, then call `initLoraAdapter` to load and attach the adapter. The adapter must be built for the same base model architecture as the GGUF you loaded.

## Key types
- `LlamaModel` — the base model; exposes `initLoraAdapter(Arena, Path)` and owns the adapter's lifecycle.
- `LlamaLoraAdapter` — thin FFM wrapper over the native `llama_adapter_lora` handle (`llama_adapter_lora_init` / `llama_adapter_lora_free`); created for you by `initLoraAdapter`.

## Usage
```java
try (Arena arena = Arena.ofConfined()) {
  // 1. Load the base model
  var modelParams = new LlamaModelParams(arena);
  var model = new LlamaModel(arena, Path.of("models/model.gguf"), modelParams);

  // 2. Load + attach a LoRA adapter (fluent: returns the same model)
  model.initLoraAdapter(arena, Path.of("models/lora-adapter.gguf").toAbsolutePath());

  // 3. Build the context and generate as usual — the adapter is now in effect
  var contextParams = new LlamaContextParams(arena)
    .nCtx(512)
    .nBatch(512);

  var context = new LlamaContext(arena, model, contextParams);
  var vocab = new LlamaVocab(model);
  var sampler = new LlamaSampler(arena).seed(42).temperature(0.75f).topK(40);
  var tokenizer = new LlamaTokenizer(vocab, context);

  var state = ConversationState.create(arena, context, tokenizer, sampler)
    .setMaxTokens(64)
    .initialize(/* tokenized prompt */ prompt);

  String output = new DefaultLlamaIterator(state)
    .stream()
    .reduce(LlamaOutput::merge)
    .orElse(new LlamaOutput("", 0))
    .content();

  // 4. Cleanup — model.free() also frees the attached adapter
  context.free();
  sampler.free();
  model.free();
}
```

## Notes
- `initLoraAdapter(Arena arena, Path loraPath)` is fluent (returns the `LlamaModel`), so it chains after construction. Call it before building the `LlamaContext` you intend to run.
- The path is resolved to an absolute path internally, but the test/`Main` pass `Path.toAbsolutePath()` explicitly — do the same to avoid surprises with the working directory.
- Lifecycle: the adapter is owned by the model. `model.free()` frees the adapter first (`LlamaLoraAdapter.free()` → `llama_adapter_lora_free`) and then frees the model — do **not** free the adapter separately.
- The `Arena` passed in must stay alive at least as long as the model, since it backs the native path string used to load the adapter.
- One adapter is tracked per model: the model holds a single `LlamaLoraAdapter` reference, so calling `initLoraAdapter` again overwrites it (the previous reference is not auto-freed). There is no separate per-context scale knob exposed by the binding.
- The adapter GGUF must match the base model (architecture and tensor shapes). In the test suite the base is `Qwen3-0.6B` with the matching `Qwen3-0.6B-tldr-lora` adapter.
- Reference usage: `Main` wires it to the `--lora <path>` CLI flag, and `TunedLlamaIteratorTest` exercises the full load-attach-generate flow.

## See also
- [Getting Started](../getting-started/README.md) — loading a model and running a first generation.
- [Text Generation & Sampling](../text-generation/README.md) — the iterator/sampler flow the adapter plugs into.
- [Devices & Memory](../device-and-memory/README.md) — Arena and native resource lifecycle.
- [Custom Builds & Platform Support](../custom-builds/README.md) — keeping the native `llama.cpp` version in sync.
