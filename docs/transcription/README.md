# Realtime Transcription

> Transcribe audio as it arrives with a streaming speech-to-text model: push PCM, get back the text heard so far.

## Overview
A streaming speech-to-text model (`MtmdCapabilities.audioStream()`) encodes the audio in small
steps and writes the transcript as it goes, so text comes back while the speaker is still talking.
`LlamaTranscriber` owns the context and the greedy sampler; the caller owns the `LlamaModel` and
the `MtmdContext` of its audio projector. Each `Session` is one live stream.

The underlying llama.cpp API is experimental: it may change between llama.cpp versions.

## Key types
- `LlamaTranscriber` — `create(arena, model, mtmd, options)`, `open()` (a `Session`), `transcribe(pcm)` (a whole recording), `sampleRate()`.
- `LlamaTranscriber.Session` — `push(pcm)` and `pause()` return the text transcribed since the previous call; `finish()` ends the audio and returns the rest; `text()` is the whole transcript; `ended()`.
- `LlamaTranscriber.Options` — `nCtx`, `nBatch`, `stepMs`.

## Usage
```java
var model = new LlamaModel(arena, Path.of("models/asr.gguf"), new LlamaModelParams(arena));
var mtmd = new MtmdContext(arena, model, Path.of("models/asr-mmproj.gguf"), new MtmdContextParams(arena));

try (var transcriber = LlamaTranscriber.create(arena, model, mtmd, LlamaTranscriber.Options.defaults());
     var session = transcriber.open()) {
    while (microphone.isOpen()) {
        System.out.print(session.push(microphone.read()));   // float PCM at transcriber.sampleRate()
    }
    System.out.print(session.finish());
}

// A whole recording, pushed as a live stream would be:
String text = transcriber.transcribe(AudioLoader.loadAudio(Path.of("call.wav"), transcriber.sampleRate()));
```

### From the CLI
```bash
java -jar llamaj.cpp-<ver>.jar --model asr.gguf --mmproj asr-mmproj.gguf --transcribe meeting.wav
```
The audio file is resampled to the model's rate and pushed as a live stream would be; the
transcript is printed.

## Options
| Option | Default | Effect |
| --- | --- | --- |
| `nCtx` | `16384` | Context size: the audio rows plus the transcript of one session. |
| `nBatch` | `512` | Batch size in tokens. |
| `stepMs` | `320` | Audio encoded per encoder call, rounded up to the model's step; also the push size of `transcribe`. |

## Notes
- Audio is mono float PCM at `sampleRate()`; resample with `AudioLoader.loadAudio(bytes, transcriber.sampleRate())`.
- `pause()` transcribes what was pushed without waiting for more audio (a pause in the speech); the session stays open. After `finish()` the session has ended and further pushes return nothing.
- One session at a time: `open()` throws until the previous session is closed. Opening a session clears the context.
- Text is returned on whole UTF-8 characters; a character split across steps comes back with the next call.
- `LlamaTranscriber` is not thread-safe. `close()` frees its context and sampler (and an open session), not the model or the `MtmdContext`.

## See also
- [Speech Generation](../speech-generation/README.md) — the reverse direction, text to speech.
- [Multimodal (Vision, Video & Audio)](../multimodal/README.md) — `MtmdContext`, capabilities, and audio files as prompt media.
