# Decision Models

> Run native decision models (classifiers that answer typed questions in one forward pass, without generating tokens) and read one raw score per option.

## Overview
A decision model is a GGUF that declares `<arch>.decision.type`. There is one decider per family,
each with its own factory and its own prompt type. Every decider:

- reads its special tokens, the temperatures and the `"systemone"` template from the GGUF
- scores prompts into **one raw score per option**

Rendering the template and turning scores into answers are left to the caller, the same way
[Reranking](../reranking/README.md) returns raw scores.

| Family | Factory | Prompt | Score of an option |
| --- | --- | --- | --- |
| `openjev`, `lev`, `nimble` | `LabelDecider.create(...)` | `TextPrompt` | logit of its label token at the last prompt token |
| any model (`CUSTOM`) | `LabelDecider.custom(..., codes)` | `TextPrompt` | logit of its caller-chosen code token at the last prompt token |
| `kev` | `KevDecider.create(...)` | `TextPrompt` with `nOptions` markers | dot product of the last token's hidden state with the hidden state at its `<\|box_end\|>` marker |
| `laya` | `LayaDecider.create(...)` | `LayaPrompt` (tokens, `QuestionType`) | the question type's column of the hidden state at its mask-token marker |
| `clef` | `ClefDecider.create(...)` | `ClefPrompt` (tokens, a `DecisionOrder` per token) | joint head output for its `DecisionOrder.OPTION` span |

Each `create` checks that the model is of its family. kev and laya find the option markers in the
prompt tokens themselves. `kev`, `laya` and `clef` read the embeddings output, so their context
runs in embedding mode, each prompt is a forward pass of its own, and a prompt must fit in one
batch.

### Typed or untyped
- Code that knows the family holds the concrete decider and calls its typed `scoreAll(List<P>)`:
  a malformed prompt does not compile.
- Code that does not holds a plain `LlamaDecider`, from `LlamaDecider.create(...)` (which picks
  the family's factory from the metadata), and calls `score(List<? extends DecisionPrompt>)`: a
  prompt of the wrong kind throws, naming the kind expected.

## Key types
- `LlamaDecider`: the sealed base class; `create(arena, model, options)`, `score(List<? extends DecisionPrompt>)`, `score(DecisionPrompt)`, `type()`, `temperatures()`, `template()`, `tokenize(text, parseSpecial)`.
- `LabelDecider`, `KevDecider`, `LayaDecider`, `ClefDecider`: one per family, each with a `create(...)` factory (and `LabelDecider.custom(...)`) and a typed `scoreAll(...)`. The family-specific accessors live here: `labels()` (label codes and their tokens), `markerToken()`, `sepToken()`, `maxHeadTokens()`.
- `DecisionPrompt`: the sealed prompt interface, with `nOptions()`. Its records:
  - `TextPrompt`: text or tokens, and `nOptions`; `TextPrompt.of(text, n)` / `TextPrompt.of(tokens, n)`.
  - `LayaPrompt`: tokens, `QuestionType`, `nOptions`.
  - `ClefPrompt`: tokens and a `DecisionOrder[]`; `nOptions()` counts the `OPTION` spans.
- `QuestionType`: `CHOICE`, `SCORE`, `NOUL`; `column()` is the laya output column.
- `DecisionOrder`: the role of a clef prompt token (`NONE`, `QUESTION_NOUL`, `QUESTION_CHOICE`, `QUESTION_SCORE`, `OPTION`).
- `DecisionType`: the family enum; `DecisionType.of(arena, model)` returns empty for an ordinary model.
- `LlamaTemplate(arena, model, name)`: loads a named template, such as `"systemone"`.

## Usage
```java
var model = new LlamaModel(arena, Path.of("models/nimble.gguf"), new LlamaModelParams(arena));

try (var decider = LabelDecider.create(arena, model, LlamaDecider.Options.defaults())) {
    String systemone = decider.template();             // render with your Jinja engine
    Map<String, Float> temps = decider.temperatures(); // raw <arch>.decision.temperature.* values

    // One rendered prompt per question; for nimble they differ only in "Requested field".
    List<float[]> scores = decider.scoreAll(List.of(
        TextPrompt.of(renderedIntent, 3),               // float[3]
        TextPrompt.of(renderedUrgency, 5)));            // float[5]
}
```

The other families follow the same pattern with their own prompt:

```java
kevDecider.scoreAll(List.of(TextPrompt.of(renderedKevPrompt, 2)));                  // 2 <|box_end|> markers
layaDecider.scoreAll(List.of(new LayaPrompt(layaTokens, QuestionType.NOUL, 2)));   // 2 mask markers
clefDecider.scoreAll(List.of(new ClefPrompt(clefTokens, order)));                  // one per OPTION span

// Family not known up front:
try (LlamaDecider decider = LlamaDecider.create(arena, model, options)) {
    decider.score(prompt);                                                         // checked at run time
}
```

For a laya prompt, build `[cls] question [sep] ([mask] option)* [sep] state [sep]`, cut to
`maxHeadTokens()`, from `LayaDecider.markerToken()` and `sepToken()`.

### Any model as a decision model
`LabelDecider.custom(arena, model, options, codes)` scores **any** causal model on caller-chosen
label tokens. The model needs no decision metadata, and `type()` is `CUSTOM`. The prompt must end
where a code is the next token, and each code must be one distinct token there (construction
checks both, tokenizing the code on its own):

```java
// A model that answers {"sentiment": "<positive|negative|neutral>"}: prefill up to the label.
try (var decider = LabelDecider.custom(arena, model, LlamaDecider.Options.defaults(),
        List.of("positive", "negative", "neutral"))) {
    decider.context().setLoraAdapter(model.loraAdapter(), 1.0f); // optional LoRA
    float[] sentiment = decider.score(TextPrompt.of(renderedChat + "{\"sentiment\": \"", 3));
}
```

Pick codes for how the prompt ends: `Answer: (` → `"A"`, `"B"`…; `Answer:` → `" Positive"` (with the
leading space); a JSON prefill → the bare word. Avoid a prompt ending in a space.

## Configuration
| Option | Default | Effect |
| --- | --- | --- |
| `nCtx` | `8192` | Context size, shared by all sequences (the KV cache is unified for label families). |
| `nBatch` | llama.cpp default (label families), `2048` (kev, laya, clef) | Tokens per decode. Label families split long prompts across decodes; for the embedding families it is also the ubatch size and the prompt limit. |
| `nSeqMax` | `8` | Label families only: sequence 0 holds the shared prefix, so up to `nSeqMax - 1` prompts decode together. `1` disables prefix sharing. |

## Notes
- **Shared prefix** (`LabelDecider`). Scoring decodes the common prefix of all prompts once in sequence 0, copies it to sequences `1..nSeqMax-1` with `LlamaMemory.seqCp`, and decodes only the tails, packed into shared batches. Each prompt keeps at least one token of its own, so its logits can be read.
- Text prompts are tokenized with special-token parsing and **without** an added BOS, as a rendered template expects. Pass tokens to tokenize yourself.
- Labels: openjev uses `A..Z a..z` and construction throws if any of them is not a single token. lev and nimble use the codes `A..Z` then `AA..ZZ` that are single tokens, up to 255. Option `i` uses `labels().get(i)`; a prompt with more options than labels is rejected.
- `ClefDecider` needs the staging symbol `llama_batch_ext_set_decision_order` (C++-mangled, bound through `LlamaExt`); check `LlamaExt.decisionAvailable()` and `decisionResolutionReport()`. In a clef prompt, spans of the same role must be separated by `NONE` tokens; a NaN score means the head could not use the order.
- Every scoring call clears the context memory first. `LlamaDecider` is not thread-safe.
- The caller owns the `LlamaModel`. `close()` frees only the decider's context. A decider is not `MemorySegmentAware`, so in tests close it with try-with-resources rather than `track()`.
- No image input.

## See also
- [Reranking](../reranking/README.md): the same raw-score wrapper pattern.
- [Chat Templates](../chat-templates/README.md): `LlamaTemplate`, including named templates.
- [Parallel Conversations](../parallel-conversations/README.md): sequence ids and the shared context.
