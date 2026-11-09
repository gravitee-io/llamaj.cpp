# Speech Generation

> Turn text into speech with a speech model and its audio-generation projector, whole or streamed.

## Overview
A speech model reads the text and produces one hidden state per audio frame; its projector (an
mmproj GGUF that generates audio, `MtmdCapabilities.audioGeneration()`) turns the frames into PCM
audio. `LlamaSpeaker` drives that loop: it owns its context and sampler, while the caller owns the
`LlamaModel` and the `MtmdContext`.

The underlying llama.cpp API is experimental: it may change between llama.cpp versions.

## Key types
- `LlamaSpeaker` — `create(arena, model, mtmd, options)`, `speak(request)`, `speak(request, onAudio)` (streaming), `sampleRate()`, `voices()`.
- `LlamaSpeaker.Request` — `of(text)`, `withVoice(name)` (a predefined voice), `withSpeakerReference(media)` (a clip to clone), `withLanguage(lang)`, `withMaxFrames(n)`.
- `LlamaSpeaker.Speech` — `sampleRate()`, `pcm()` (mono float samples in `[-1, 1]`), `frames()`, `seconds()`, `wav()` (16-bit mono WAV bytes).
- `LlamaSpeaker.Options` — `nCtx`, `nBatch`, `temperature`, `topK`, `topP`, `seed`.

## Usage
```java
var model = new LlamaModel(arena, Path.of("models/tts.gguf"), new LlamaModelParams(arena));
var mtmd = new MtmdContext(arena, model, Path.of("models/tts-mmproj.gguf"), new MtmdContextParams(arena));

try (var speaker = LlamaSpeaker.create(arena, model, mtmd, LlamaSpeaker.Options.defaults())) {
    var speech = speaker.speak(LlamaSpeaker.Request.of("The weather in Paris is sunny today."));
    Files.write(Path.of("weather.wav"), speech.wav());

    // Streamed: each new chunk of PCM samples as it is produced
    speaker.speak(LlamaSpeaker.Request.of("Streaming works too."), chunk -> player.write(chunk));
}
```

### From the CLI
```bash
java -jar llamaj.cpp-<ver>.jar --model tts.gguf --mmproj tts-mmproj.gguf \
  --speak "Hello there" --speech_out hello.wav [--voice <name>]
```
It prints the model's voices, then writes a 16-bit mono WAV file (`speech.wav` by default).

## Options
| Option | Default | Effect |
| --- | --- | --- |
| `nCtx` | `4096` | Context size in tokens. |
| `nBatch` | `512` | Batch and ubatch size; prompt tokens processed per step. |
| `temperature` | `0.9` | Sampling temperature of the frame codes. |
| `topK` | `50` | Top-k of the frame codes. |
| `topP` | `1.0` | Top-p of the frame codes. |
| `seed` | random | Sampling seed; fix it for reproducible audio. |

`Request.maxFrames` (default `512`) caps the frames generated; the model stops earlier at the end of speech.

## Notes
- A model with predefined voices speaks with `voices().getFirst()` unless the request names another voice or gives a speaker reference; a request gives a voice or a reference, not both.
- A speaker reference is an audio clip (`MtmdAudio`) loaded at `mtmd.getAudioSampleRate()`.
- `speak` resets the generation state and clears the context first, so requests are independent.
- `LlamaSpeaker` is not thread-safe. `close()` frees its context and sampler, not the model or the `MtmdContext`.

## See also
- [Multimodal (Vision, Video & Audio)](../multimodal/README.md) — `MtmdContext`, capabilities, and audio input.
