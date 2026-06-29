# Multimodal (Vision & Audio)

> Feed images and audio to a vision/audio-capable model by pairing the main LLM with an mmproj (clip) projector through `MtmdContext`.

## Overview
Multimodal support lets you attach images (JPEG, PNG, GIF, BMP, WebP, TIFF) and audio (WAV) to a prompt so a vision/audio model can reason over them. Media is decoded into native `mtmd_bitmap` handles, attached to a `ConversationState` via `setMedia`, and tokenized/encoded by a `MtmdContext` that wraps the model's separate **mmproj** (clip/projector) GGUF. You drive generation with the same `DefaultLlamaIterator` as text, but constructed with the `MtmdContext`. Use it whenever your model ships a companion mmproj file and you need to pass non-text inputs.

## Key types
- `MtmdContext` — wraps the native `mtmd_context`; loads the mmproj projector and exposes `supportsVision()`, `supportsAudio()`, `getAudioSampleRate()`. Required for any multimodal generation.
- `MtmdContextParams` — fluent config for the context (`useGpu`, `mediaMarker`, `nThreads`, `printTimings`, `flashAttnType`, image token bounds).
- `MtmdMedia` — common `Freeable` interface for media items; backs each media marker in the prompt.
- `MtmdImage` — image media; factories `fromFile`, `fromBytes`, `fromBufferedImage`, `fromBytesNative`.
- `MtmdAudio` — audio media; factories `fromFile`, `fromBytes`, `fromSamples` (all take a target sample rate).
- `MtmdMediaFactory` — auto-detects and loads media via built-in handlers (`from`, `fromFile`).
- `MediaHandler<T>` / `ImageHandler` / `AudioHandler` — pluggable loaders; image is the fallback, audio matches WAV by extension/magic bytes.
- `AudioLoader` / `AudioFormat` — Java Sound decoding + resampling and WAV/MP3/FLAC detection.
- `ConversationState.setMedia(List<MtmdMedia>)` — attaches loaded media to the prompt (one item per media marker).

## Usage
```java
Arena arena = Arena.ofShared();
LlamaRuntime.llama_backend_init();

// 1. Load the main vision model
var modelParams = new LlamaModelParams(arena);
var llamaModel = new LlamaModel(arena, mainModelPath, modelParams);

// 2. Build the multimodal context from the mmproj (clip) projector GGUF
var mtmdParams = new MtmdContextParams(arena)
  .useGpu(true)
  .mediaMarker("<IMG>")
  .printTimings(true);
var mtmdContext = new MtmdContext(arena, llamaModel, mmprojPath.toAbsolutePath(), mtmdParams);

// 3. Regular inference context
var ctxParams = new LlamaContextParams(arena).noPerf(false);
var llamaContext = new LlamaContext(arena, llamaModel, ctxParams);

// 4. Load an image (audio: MtmdAudio.fromFile(arena, path, mtmdContext.getAudioSampleRate()))
var image = MtmdImage.fromFile(arena, imagePath);

// 5. Build the conversation; the prompt must contain the media marker ("<IMG>")
var vocab = new LlamaVocab(llamaModel);
var tokenizer = new LlamaTokenizer(vocab, llamaContext);
var sampler = new LlamaSampler(arena).greedy().seed(42);

var state = ConversationState.create(arena, llamaContext, tokenizer, sampler)
  .initialize("USER: What is in this image?\n<IMG>\nASSISTANT:")
  .setMaxTokens(256)
  .setMedia(List.of(image));

// 6. Drive generation with the MtmdContext-aware iterator
var it = new DefaultLlamaIterator(state, mtmdContext);
String output = it.stream()
  .reduce(LlamaOutput::merge)
  .orElse(new LlamaOutput("", 0))
  .content();

// 7. Free native resources
image.free();
mtmdContext.free();
llamaContext.free();
sampler.free();
llamaModel.free();
```

## Options
`MtmdContextParams` (fluent setters return `this`):

| Method | Default behavior | Purpose |
|---|---|---|
| `useGpu(boolean)` | model default | Offload the projector to GPU. |
| `mediaMarker(String)` | native default marker | Placeholder token in the prompt where each media item is spliced in (e.g. `"<IMG>"`). |
| `nThreads(int)` | native default | CPU threads for the projector. |
| `printTimings(boolean)` | false | Print encode timings. |
| `flashAttnType(FlashAttentionType)` | `AUTO` | Flash-attention mode: `AUTO`, `DISABLED`, `ENABLED`. |
| `imageMinTokens(int)` / `imageMaxTokens(int)` | model default | Clamp per-image token budget. |

`AudioFormat`: `WAV` (decoded natively), `MP3`, `FLAC` (detected only — convert to WAV before loading).

## Notes
- An mmproj/clip GGUF plus a constructed `MtmdContext` is **mandatory**; without it the iterator runs as plain text. Pass the `MtmdContext` to `new DefaultLlamaIterator(state, mtmdContext)`.
- The prompt must include the media marker once per attached item; the number of markers must match `setMedia(...)` list size.
- `setImages(List<MtmdImage>)` is deprecated — use `setMedia(List<MtmdMedia>)`, which accepts both images and audio.
- Audio must be loaded at the model's sample rate: pass `mtmdContext.getAudioSampleRate()` (e.g. 16000 for Whisper-style encoders). Only WAV decodes natively via Java Sound; convert MP3/FLAC first.
- Images are decoded to RGB via `ImageIO`; `fromBytesNative` instead uses the native stb_image decoder for byte-for-byte parity with the reference llama.cpp server.
- Each `MtmdImage`/`MtmdAudio`/`MtmdContext` is `Freeable` — call `free()` (or track them in tests) so native `mtmd_bitmap`/context memory is released before JVM exit. Allocate with an `Arena` you control.
- Check capabilities before loading: `mtmdContext.supportsVision()` / `supportsAudio()`; `AudioHandler` throws if the model lacks audio support.
- `MtmdMediaFactory.from(arena, input, mtmdContext)` auto-selects a handler (audio first, image fallback) for `Path` or `byte[]` inputs; supply custom `MediaHandler`s for other sources.

## See also
- [Getting Started](../getting-started/README.md) — model/context/arena setup this builds on.
- [Text Generation & Sampling](../text-generation/README.md) — the iterator + sampler loop reused here.
- [Chat Templates](../chat-templates/README.md) — formatting prompts (and placing media markers) for chat models.
- [Devices & Memory](../device-and-memory/README.md) — GPU offload and arena/lifecycle details.
