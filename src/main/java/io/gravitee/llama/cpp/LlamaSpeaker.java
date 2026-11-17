/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.llama.cpp;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.stream.IntStream;

/**
 * Text-to-speech with a speech model and its audio-generation projector: the model reads the text
 * and produces one hidden state per audio frame, and the projector turns the frames into PCM audio.
 * <p>
 * The speaker owns its context and sampler; the caller owns the {@link LlamaModel} and the
 * {@link MtmdContext}, which must generate speech ({@link MtmdCapabilities#audioGeneration()}).
 * The underlying llama.cpp API is experimental and may change between versions.
 *
 * <pre>{@code
 * try (var speaker = LlamaSpeaker.create(arena, model, mtmd, LlamaSpeaker.Options.defaults())) {
 *   var speech = speaker.speak(LlamaSpeaker.Request.of("Hello there."));
 *   Files.write(Path.of("hello.wav"), speech.wav());
 * }
 * }</pre>
 *
 * <h2>Thread safety</h2>
 * Not thread-safe.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaSpeaker implements Freeable, AutoCloseable {

  private static final int OUTTYPE_PCM = 0;

  /**
   * Configuration. Any {@code null} field takes a default.
   *
   * @param nCtx        Context size in tokens; {@code null} -> 4096
   * @param nBatch      Batch and ubatch size in tokens; {@code null} -> 512
   * @param temperature Sampling temperature of the frame codes; {@code null} -> 0.9
   * @param topK        Top-k of the frame codes; {@code null} -> 50
   * @param topP        Top-p of the frame codes; {@code null} -> 1.0
   * @param seed        Sampling seed; {@code null} -> random
   */
  public record Options(
    Integer nCtx,
    Integer nBatch,
    Float temperature,
    Integer topK,
    Float topP,
    Integer seed
  ) {
    public static Options defaults() {
      return new Options(null, null, null, null, null, null);
    }

    public Options withNCtx(int nCtx) {
      return new Options(nCtx, nBatch, temperature, topK, topP, seed);
    }

    public Options withNBatch(int nBatch) {
      return new Options(nCtx, nBatch, temperature, topK, topP, seed);
    }

    public Options withTemperature(float temperature) {
      return new Options(nCtx, nBatch, temperature, topK, topP, seed);
    }

    public Options withTopK(int topK) {
      return new Options(nCtx, nBatch, temperature, topK, topP, seed);
    }

    public Options withTopP(float topP) {
      return new Options(nCtx, nBatch, temperature, topK, topP, seed);
    }

    public Options withSeed(int seed) {
      return new Options(nCtx, nBatch, temperature, topK, topP, seed);
    }
  }

  /**
   * What to say, and with which voice.
   *
   * @param text             The text to speak
   * @param voice            A predefined voice of the model ({@link #voices()}), or {@code null}
   * @param speakerReference A clip of the voice to clone (an {@link MtmdAudio}), or {@code null}
   * @param language         The language of the text, for models that take one, or {@code null}
   * @param maxFrames        Most audio frames to generate; {@code null} -> 512
   */
  public record Request(
    String text,
    String voice,
    MtmdMedia speakerReference,
    String language,
    Integer maxFrames
  ) {
    public Request {
      Objects.requireNonNull(text, "text");
      if (voice != null && speakerReference != null) {
        throw new IllegalArgumentException(
          "give a voice or a speaker reference, not both"
        );
      }
    }

    public static Request of(String text) {
      return new Request(text, null, null, null, null);
    }

    public Request withVoice(String voice) {
      return new Request(text, voice, null, language, maxFrames);
    }

    public Request withSpeakerReference(MtmdMedia speakerReference) {
      return new Request(text, null, speakerReference, language, maxFrames);
    }

    public Request withLanguage(String language) {
      return new Request(text, voice, speakerReference, language, maxFrames);
    }

    public Request withMaxFrames(int maxFrames) {
      return new Request(text, voice, speakerReference, language, maxFrames);
    }
  }

  /**
   * Generated speech: mono float PCM in {@code [-1, 1]}.
   *
   * @param sampleRate The sample rate in Hz
   * @param pcm        The samples
   * @param frames     The audio frames the model generated
   */
  public record Speech(int sampleRate, float[] pcm, int frames) {
    /** The duration of the audio in seconds. */
    public double seconds() {
      return sampleRate > 0 ? (double) pcm.length / sampleRate : 0;
    }

    /** The audio as a 16-bit mono WAV file. */
    public byte[] wav() {
      int dataSize = pcm.length * 2;
      var buf = ByteBuffer.allocate(44 + dataSize).order(
        ByteOrder.LITTLE_ENDIAN
      );
      buf.put("RIFF".getBytes()).putInt(36 + dataSize).put("WAVE".getBytes());
      buf
        .put("fmt ".getBytes())
        .putInt(16)
        .putShort((short) 1)
        .putShort((short) 1)
        .putInt(sampleRate)
        .putInt(sampleRate * 2)
        .putShort((short) 2)
        .putShort((short) 16);
      buf.put("data".getBytes()).putInt(dataSize);
      for (float sample : pcm) {
        float clamped = Math.max(-1f, Math.min(1f, sample));
        buf.putShort((short) Math.round(clamped * 32767f));
      }
      return buf.array();
    }
  }

  private final MtmdContext mtmd;
  private final LlamaContext context;
  private final LlamaSampler sampler;
  private final Options options;
  private final int nBatch;
  private final int sampleRate;
  private final List<String> voices;
  private MemorySegment gen;

  private LlamaSpeaker(
    Arena arena,
    LlamaModel model,
    MtmdContext mtmd,
    Options options
  ) {
    this.mtmd = mtmd;
    this.options = options;
    this.nBatch = options.nBatch() != null ? options.nBatch() : 512;
    try (var local = Arena.ofConfined()) {
      MemorySegment info = LlamaRuntime.mtmd_gen_audio_get_info(
        local,
        mtmd.segment
      );
      if (LlamaRuntime.mtmd_gen_audio_info_field(info, "type") == 0) {
        throw new LlamaException("the projector does not generate speech");
      }
      this.sampleRate = LlamaRuntime.mtmd_gen_audio_info_field(
        info,
        "sample_rate"
      );
      int nVoices = LlamaRuntime.mtmd_gen_audio_info_field(info, "n_voices");
      this.voices = IntStream.range(0, nVoices)
        .mapToObj(i -> LlamaRuntime.mtmd_gen_audio_voice_name(mtmd.segment, i))
        .filter(Objects::nonNull)
        .toList();
    }
    this.context = new LlamaContext(
      arena,
      model,
      new LlamaContextParams(arena)
        .nCtx(options.nCtx() != null ? options.nCtx() : 4096)
        .nBatch(nBatch)
        .nUBatch(nBatch)
        .embeddings(true)
        .nSeqMax(2)
        .kvUnified(true)
    );
    try {
      this.sampler = new LlamaSampler(arena)
        .topK(options.topK() != null ? options.topK() : 50)
        .topP(options.topP() != null ? options.topP() : 1.0f, 1)
        .temperature(
          options.temperature() != null ? options.temperature() : 0.9f
        )
        .seed(options.seed() != null ? options.seed() : -1);
      this.gen = LlamaRuntime.mtmd_helper_gen_audio_init(
        context.segment,
        mtmd.segment
      );
      if (gen == null || gen.address() == 0) {
        throw new LlamaException("mtmd_helper_gen_audio_init failed");
      }
    } catch (RuntimeException e) {
      context.free();
      throw e;
    }
  }

  /**
   * Creates a speaker.
   *
   * @param arena   Arena that owns the speaker's context and sampler
   * @param model   The speech model (not freed on {@link #close()})
   * @param mtmd    Its audio-generation projector (not freed on {@link #close()})
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @throws LlamaException if the projector does not generate speech
   */
  public static LlamaSpeaker create(
    Arena arena,
    LlamaModel model,
    MtmdContext mtmd,
    Options options
  ) {
    return new LlamaSpeaker(arena, model, mtmd, options);
  }

  /** The sample rate of the generated audio, in Hz. */
  public int sampleRate() {
    checkNotFreed();
    return sampleRate;
  }

  /** The predefined voices of the model; empty if it has none. */
  public List<String> voices() {
    checkNotFreed();
    return voices;
  }

  /**
   * Speaks the text of a request.
   *
   * @param request What to say
   * @return The speech
   */
  public Speech speak(Request request) {
    return speak(request, null);
  }

  /**
   * Speaks the text of a request, handing the audio to {@code onAudio} as it is produced.
   *
   * @param request What to say
   * @param onAudio Receives each new chunk of PCM samples, or {@code null} for no streaming
   * @return The whole speech
   */
  public Speech speak(Request request, Consumer<float[]> onAudio) {
    checkNotFreed();
    boolean stream = onAudio != null;
    int maxFrames = request.maxFrames() != null ? request.maxFrames() : 512;
    LlamaRuntime.mtmd_helper_gen_audio_reset(gen);
    context.getMemory().clear();

    MemorySegment voiceBitmap = null;
    try (var local = Arena.ofConfined()) {
      MemorySegment speakerRef = MemorySegment.NULL;
      if (request.speakerReference() != null) {
        speakerRef = request.speakerReference().getMemorySegment();
      } else if (request.voice() != null || !voices.isEmpty()) {
        String voice = request.voice() != null
          ? request.voice()
          : voices.getFirst();
        voiceBitmap = LlamaRuntime.mtmd_gen_audio_get_voice(
          mtmd.segment,
          local.allocateFrom(voice)
        );
        if (voiceBitmap == null || voiceBitmap.address() == 0) {
          throw new LlamaException(
            "unknown voice \"%s\", the model has %s".formatted(voice, voices)
          );
        }
        speakerRef = voiceBitmap;
      }

      MemorySegment inp = LlamaRuntime.mtmd_helper_gen_audio_inp_default(local);
      MemorySegment prompt = local.allocateFrom(request.text());
      set(inp, "stream", boolean.class, stream);
      set(inp, "seq_id", int.class, 0);
      set(inp, "prompt", MemorySegment.class, prompt);
      set(inp, "prompt_len", long.class, prompt.byteSize() - 1);
      set(inp, "speaker_ref", MemorySegment.class, speakerRef);
      if (request.language() != null) {
        set(
          inp,
          "lang",
          MemorySegment.class,
          local.allocateFrom(request.language())
        );
      }
      if (options.topK() != null) set(inp, "top_k", int.class, options.topK());
      if (options.topP() != null) set(
        inp,
        "top_p",
        float.class,
        options.topP()
      );
      if (options.seed() != null) set(inp, "seed", int.class, options.seed());
      set(inp, "out_type", int.class, OUTTYPE_PCM);

      if (LlamaRuntime.mtmd_helper_gen_audio_set_input(gen, inp) != 0) {
        throw new LlamaException("mtmd_helper_gen_audio_set_input failed");
      }
      int remaining;
      do {
        remaining = LlamaRuntime.mtmd_helper_gen_audio_step_prompt(gen, nBatch);
        if (remaining < 0) {
          throw new LlamaException("speech prompt processing failed");
        }
      } while (remaining > 0);

      var chunks = new ArrayList<float[]>();
      MemorySegment hOut = local.allocate(ADDRESS);
      MemorySegment stop = local.allocate(JAVA_BOOLEAN);
      int sampled = sampler.sample(context);
      MemorySegment hState = LlamaRuntime.llama_get_embeddings_ith(
        context.segment,
        -1
      );
      int frames = 0;
      while (frames < maxFrames) {
        hOut.set(ADDRESS, 0, MemorySegment.NULL);
        stop.set(JAVA_BOOLEAN, 0, false);
        if (
          LlamaRuntime.mtmd_helper_gen_audio_step_gen(
            gen,
            sampled,
            hState,
            hOut,
            stop
          ) !=
          0
        ) {
          throw new LlamaException(
            "speech generation failed at frame %d".formatted(frames)
          );
        }
        MemorySegment hNext = hOut.get(ADDRESS, 0);
        if (hNext.address() == 0) {
          break;
        }
        frames++;
        if (stream) {
          emit(local, chunks, onAudio);
        }
        if (stop.get(JAVA_BOOLEAN, 0)) {
          break;
        }
        hState = hNext;
        sampled = sampler.sample(context);
      }
      if (stream) {
        LlamaRuntime.mtmd_helper_gen_audio_flush(gen);
      }
      emit(local, chunks, onAudio);
      return new Speech(sampleRate, concat(chunks), frames);
    } finally {
      if (voiceBitmap != null && voiceBitmap.address() != 0) {
        LlamaRuntime.mtmd_bitmap_free(voiceBitmap);
      }
    }
  }

  private void emit(Arena local, List<float[]> chunks, Consumer<float[]> sink) {
    MemorySegment rate = local.allocate(JAVA_INT);
    MemorySegment data = local.allocate(ADDRESS);
    MemorySegment len = local.allocate(JAVA_LONG);
    MemorySegment n = local.allocate(JAVA_LONG);
    if (
      LlamaRuntime.mtmd_helper_gen_audio_get_output(gen, rate, data, len, n) !=
      0
    ) {
      throw new LlamaException("mtmd_helper_gen_audio_get_output failed");
    }
    long bytes = len.get(JAVA_LONG, 0);
    if (bytes <= 0) {
      return;
    }
    float[] pcm = data.get(ADDRESS, 0).reinterpret(bytes).toArray(JAVA_FLOAT);
    chunks.add(pcm);
    if (sink != null) {
      sink.accept(pcm);
    }
  }

  private static float[] concat(List<float[]> chunks) {
    int total = chunks
      .stream()
      .mapToInt(c -> c.length)
      .sum();
    float[] pcm = new float[total];
    int offset = 0;
    for (float[] chunk : chunks) {
      System.arraycopy(chunk, 0, pcm, offset, chunk.length);
      offset += chunk.length;
    }
    return pcm;
  }

  private static void set(
    MemorySegment inp,
    String field,
    Class<?> type,
    Object value
  ) {
    LlamaRuntime.mtmd_helper_gen_audio_inp_set(inp, field, type, value);
  }

  /** Exposes the underlying context for advanced use cases. */
  public LlamaContext context() {
    checkNotFreed();
    return context;
  }

  @Override
  public void free() {
    if (gen == null) return;
    LlamaRuntime.mtmd_helper_gen_audio_free(gen);
    gen = null;
    sampler.free();
    context.free();
  }

  @Override
  public boolean isFree() {
    return gen == null;
  }

  @Override
  public void close() {
    free();
  }

  private void checkNotFreed() {
    if (gen == null) {
      throw new LlamaException("LlamaSpeaker has been freed");
    }
  }
}
