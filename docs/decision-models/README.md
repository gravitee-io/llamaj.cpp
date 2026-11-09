# Decision Models

> Run native decision models (classifiers that answer typed questions in one forward pass, without generating tokens) and read one raw score per option.

## Overview
A decision model is a GGUF that declares `<arch>.decision.type`. There is one decider per family,
each with its own factory and its own prompt type. Every decider:

- reads its special tokens, the temperatures, the score biases and the `"systemone"` template from the GGUF
- scores prompts into **one raw score per option**

Rendering the template and turning scores into answers are left to the caller, the same way
[Reranking](../reranking/README.md) returns raw scores.

| Family | Factory | Prompt | Score of an option |
| --- | --- | --- | --- |
| `openjev`, `lev`, `nimble`, `pplx-decider`, `vllm-d3`, `granite-switch`, `h2o-lightning` | `LabelDecider.create(...)` | `TextPrompt`, `MediaPrompt` | logit of its label token at the last prompt token |
| any model (`CUSTOM`) | `LabelDecider.custom(..., context, codes)` | `TextPrompt`, `MediaPrompt` | logit of its caller-chosen code token at the last prompt token |
| `kev` | `KevDecider.create(...)` | `TextPrompt` with `nOptions` markers | dot product of the last token's hidden state with the hidden state at its `<\|box_end\|>` marker |
| `laya` | `LayaDecider.create(...)` | `LayaPrompt` (tokens, `QuestionType`) | the question type's column of the hidden state at its mask-token marker |
| `clef`, `vllm-decision2` | `ClefDecider.create(...)` | `ClefPrompt` (tokens, a `DecisionOrder` per token) | joint head output for its `DecisionOrder.OPTION` span |
| `openbmb-minicpm-ocr-jev` | `PooledDecider.create(...)` | `TextPrompt`, `MediaPrompt` | one score per class (`classes()`), the pooled output of the sequence |

Each `create` checks that the model is of its family. kev and laya find the option markers in the
prompt tokens themselves. `kev`, `laya`, `clef`, `vllm-decision2` and `openbmb-minicpm-ocr-jev`
read the embeddings output, so their context runs in embedding mode, each prompt is a forward pass
of its own, and a text prompt must fit in one batch. A clef prompt holds all the questions of a
request, a vllm-decision2 prompt one question.

### Typed or untyped
- Code that knows the family holds the concrete decider and calls its typed `scoreAll(List<P>)`:
  a malformed prompt does not compile.
- Code that does not holds a plain `LlamaDecider`, from `LlamaDecider.create(...)` (which picks
  the family's factory from the metadata), and calls `score(List<? extends DecisionPrompt>)`: a
  prompt of the wrong kind throws, naming the kind expected.

## Key types
- `LlamaDecider`: the sealed base class; `create(arena, model, options)`, `create(arena, model, mtmd, options)`, `score(List<? extends DecisionPrompt>)`, `score(DecisionPrompt)`, `type()`, `temperatures()`, `scoreBiases()`, `causal()`, `template()`, `tokenize(text, parseSpecial)`.
- `LabelDecider`, `KevDecider`, `LayaDecider`, `ClefDecider`, `PooledDecider`: one per mechanism, each with a `create(...)` factory (and `LabelDecider.custom(...)`) and a typed `scoreAll(...)` (plus `scoreMedia(...)` for `LabelDecider` and `PooledDecider`). The family-specific accessors live here: `labels()` (label codes and their tokens), `markerToken()`, `sepToken()`, `maxHeadTokens()`, `classes()`.
- `DecisionPrompt`: the sealed prompt interface, with `nOptions()`. Its records:
  - `TextPrompt`: text or tokens, and `nOptions`; `TextPrompt.of(text, n)` / `TextPrompt.of(tokens, n)`.
  - `MediaPrompt`: rendered text with one media marker (`MtmdContext.marker()`) per image or audio item, the media, and `nOptions`; `MediaPrompt.of(text, media, n)`.
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
`LabelDecider.custom(...)` scores **any** causal model on caller-chosen label tokens. The model
needs no decision metadata, and `type()` is `CUSTOM`. The prompt must end where a code is the next
token, and each code must be one distinct token there. Give the decider the text the prompts end
with, and it resolves each code in that context:

```java
// A model that answers {"sentiment": "<positive|negative|neutral>"}: prefill up to the label.
String prefill = "{\"sentiment\": \"";
try (var decider = LabelDecider.custom(arena, model, LlamaDecider.Options.defaults(),
        prefill, List.of("positive", "negative", "neutral"))) {
    decider.context().setLoraAdapter(model.loraAdapter(), 1.0f); // optional LoRA
    float[] sentiment = decider.score(TextPrompt.of(renderedChat + prefill, 3));
}
```

Each code's token is the one `context + code` adds after the tokens of `context`; construction
fails if a code adds more than one token there, or if two codes share a token.
`custom(arena, model, mtmd, options, context, codes)` does the same for media prompts.
`custom(arena, model, options, codes)`, without the context, tokenizes each code on its own. That
is the same token on BPE vocabs (Qwen, Llama 3, Gemma), but on SentencePiece vocabs with
`add_space_prefix` (Llama 2, Mistral) a standalone `"A"` becomes `"▁A"`, which is not what the
model predicts after `Answer: (`. Prefer the context form.

Pick codes for how the prompt ends: `Answer: (` → `"A"`, `"B"`…; a JSON prefill → the bare word.
Avoid a prompt ending in a space.

### Images and audio
`LabelDecider` and `PooledDecider` score `MediaPrompt`s when they are created with the model's
`MtmdContext`. Render the template with `mtmd.marker()` where each image or audio item goes, then
pass the media in marker order:

```java
var mtmd = new MtmdContext(arena, model, Path.of("models/mmproj.gguf"), new MtmdContextParams(arena));
try (var decider = LlamaDecider.create(arena, model, mtmd, LlamaDecider.Options.defaults())) {
    var image = MtmdImage.fromFile(arena, Path.of("scan.png"));
    float[] scores = decider.score(MediaPrompt.of(renderedWithMarker, List.of(image), 2));
}
```

Each media prompt is a forward pass of its own (no prefix sharing). A media prompt is rejected by
a decider created without an `MtmdContext`, and by a non-causal model.

### From the CLI
`--decide <prompt>` scores one prompt and prints a raw score per option, named by the model's
labels or classes when it has them; `--options <n>` sets the option count otherwise. Add
`--mmproj` and `--media` to score a media prompt. The prompt is passed as is, so write it in the
family's format:

```bash
java -jar llamaj.cpp-<ver>.jar --model decision.gguf --decide "<prompt in the model's format>"
java -jar llamaj.cpp-<ver>.jar --model ocr.gguf --mmproj ocr-mmproj.gguf --media page.png \
  --decide "<__media__>…"
```

## Configuration
| Option | Default | Effect |
| --- | --- | --- |
| `nCtx` | `8192` | Context size, shared by all sequences (the KV cache is unified for label families). |
| `nBatch` | llama.cpp default (causal label families), `2048` (the others) | Tokens per decode. Causal label families split long prompts across decodes; for the other families it is also the ubatch size and the text-prompt limit. |
| `nSeqMax` | `8` | Causal label families only: sequence 0 holds the shared prefix, so up to `nSeqMax - 1` prompts decode together. `1` disables prefix sharing. |

## Notes
- **Shared prefix** (`LabelDecider`). Scoring decodes the common prefix of all prompts once in sequence 0, copies it to sequences `1..nSeqMax-1` with `LlamaMemory.seqCp`, and decodes only the tails, packed into shared batches. Each prompt keeps at least one token of its own, so its logits can be read.
- Text prompts are tokenized with special-token parsing and **without** an added BOS, as a rendered template expects. Pass tokens to tokenize yourself.
- **Non-causal models** (`causal()` is false, from `<arch>.attention.causal`, e.g. `vllm-d3`). Every token reads the whole prompt, so there is no prefix to share: each prompt is decoded on its own in one ubatch and must fit in `nBatch` (2048 by default). They take text prompts only.
- Labels: openjev uses `A..Z a..z` and construction throws if any of them is not a single token. lev, nimble, pplx-decider, vllm-d3 and granite-switch use the codes `A..Z` then `AA..ZZ` that are single tokens, up to 255. h2o-lightning reads its label texts from the GGUF (`<arch>.decision.labels`, e.g. `" A"`), each a single distinct token; that needs the staging symbol `llama_model_decision_label` (`LlamaExt.decisionLabelsAvailable()`). Option `i` uses `labels().get(i)`; a prompt with more options than labels is rejected.
- `scoreBiases()` returns the `<arch>.decision.score_bias.<n>.<level>` offsets keyed by the number of levels `n`: add `scoreBiases().get(n)[i]` to the raw score of level `i` of an `n`-level score question before the softmax.
- `PooledDecider` scores every prompt into all the classes of the model: `nOptions` must be `classes().size()`.
- `ClefDecider` needs the staging symbol `llama_batch_ext_set_decision_order` (C++-mangled, bound through `LlamaExt`); check `LlamaExt.decisionAvailable()` and `decisionResolutionReport()`. In a clef prompt, spans of the same role must be separated by `NONE` tokens; a NaN score means the head could not use the order.
- Every scoring call clears the context memory first. `LlamaDecider` is not thread-safe.
- The caller owns the `LlamaModel`. `close()` frees only the decider's context. A decider is not `MemorySegmentAware`, so in tests close it with try-with-resources rather than `track()`.
- Image and audio input: `LabelDecider` and `PooledDecider` only (`MediaPrompt`); kev, laya, clef and vllm-decision2 take text.

## See also
- [Reranking](../reranking/README.md): the same raw-score wrapper pattern.
- [Chat Templates](../chat-templates/README.md): `LlamaTemplate`, including named templates.
- [Parallel Conversations](../parallel-conversations/README.md): sequence ids and the shared context.
