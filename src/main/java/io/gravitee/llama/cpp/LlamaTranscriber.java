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

import static io.gravitee.llama.cpp.LlamaRuntime.LLAMA_PROCESS_TYPE_DECODE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.io.ByteArrayOutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * Realtime transcription with a streaming speech-to-text model and its audio projector: audio is
 * pushed as it arrives, and the text of the audio heard so far comes back after each push.
 * <p>
 * The transcriber owns its context and sampler; the caller owns the {@link LlamaModel} and the
 * {@link MtmdContext}, which must transcribe streams ({@link MtmdCapabilities#audioStream()}).
 * Audio is mono float PCM at {@link #sampleRate()}. One {@link Session} at a time. The underlying
 * llama.cpp API is experimental and may change between versions.
 *
 * <pre>{@code
 * try (var transcriber = LlamaTranscriber.create(arena, model, mtmd, LlamaTranscriber.Options.defaults());
 *      var session = transcriber.open()) {
 *   while (microphone.hasMore()) {
 *     System.out.print(session.push(microphone.read()));
 *   }
 *   System.out.print(session.finish());
 * }
 * }</pre>
 *
 * <h2>Thread safety</h2>
 * Not thread-safe.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaTranscriber implements Freeable, AutoCloseable {

  private static final int TOKEN_NULL = -1;
  private static final int POSITIONS_PER_ROW = 4;

  /**
   * Configuration. Any {@code null} field takes a default.
   *
   * @param nCtx   Context size in tokens, the longest transcript (with the audio rows) of a
   *               session; {@code null} -> 16384
   * @param nBatch Batch size in tokens; {@code null} -> 512
   * @param stepMs Audio encoded per encoder call, rounded up to the model step; {@code null} ->
   *               llama.cpp's default (320)
   */
  public record Options(Integer nCtx, Integer nBatch, Integer stepMs) {
    public static Options defaults() {
      return new Options(null, null, null);
    }

    public Options withNCtx(int nCtx) {
      return new Options(nCtx, nBatch, stepMs);
    }

    public Options withNBatch(int nBatch) {
      return new Options(nCtx, nBatch, stepMs);
    }

    public Options withStepMs(int stepMs) {
      return new Options(nCtx, nBatch, stepMs);
    }
  }

  private final MtmdContext mtmd;
  private final LlamaContext context;
  private final LlamaSampler sampler;
  private final LlamaVocab vocab;
  private final Options options;
  private final int nEmbd;
  private final int sampleRate;
  private Session open;
  private boolean freed;

  private LlamaTranscriber(
    Arena arena,
    LlamaModel model,
    MtmdContext mtmd,
    Options options
  ) {
    if (!mtmd.capabilities().audioStream()) {
      throw new LlamaException("the model does not transcribe audio streams");
    }
    this.mtmd = mtmd;
    this.options = options;
    this.vocab = new LlamaVocab(model);
    this.nEmbd = LlamaRuntime.llama_model_n_embd_inp(model.segment);
    this.sampleRate = mtmd.getAudioSampleRate();
    int nBatch = options.nBatch() != null ? options.nBatch() : 512;
    this.context = new LlamaContext(
      arena,
      model,
      new LlamaContextParams(arena)
        .nCtx(options.nCtx() != null ? options.nCtx() : 16384)
        .nBatch(nBatch)
        .nSeqMax(1)
    );
    this.sampler = new LlamaSampler(arena).greedy();
  }

  /**
   * Creates a transcriber.
   *
   * @param arena   Arena that owns the transcriber's context and sampler
   * @param model   The streaming speech-to-text model (not freed on {@link #close()})
   * @param mtmd    Its audio projector (not freed on {@link #close()})
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @throws LlamaException if the model does not transcribe audio streams
   */
  public static LlamaTranscriber create(
    Arena arena,
    LlamaModel model,
    MtmdContext mtmd,
    Options options
  ) {
    return new LlamaTranscriber(arena, model, mtmd, options);
  }

  /** The sample rate of the audio to push, in Hz. */
  public int sampleRate() {
    checkNotFreed();
    return sampleRate;
  }

  /**
   * Opens a session, clearing the context. The previous session must have been closed.
   *
   * @throws LlamaException if a session is open, or the stream cannot be created
   */
  public Session open() {
    checkNotFreed();
    if (open != null) {
      throw new LlamaException("a session is already open");
    }
    open = new Session();
    return open;
  }

  /**
   * Transcribes a whole recording, pushed in steps of {@link Options#stepMs()} as a live stream
   * would be.
   *
   * @param pcm Mono float PCM at {@link #sampleRate()}
   * @return The transcript
   */
  public String transcribe(float[] pcm) {
    try (var session = open()) {
      int step = Math.max(
        1,
        (sampleRate * (options.stepMs() != null ? options.stepMs() : 320)) /
          1000
      );
      for (int from = 0; from < pcm.length; from += step) {
        session.push(
          Arrays.copyOfRange(pcm, from, Math.min(pcm.length, from + step))
        );
      }
      session.finish();
      return session.text();
    }
  }

  /** Exposes the underlying context for advanced use cases. */
  public LlamaContext context() {
    checkNotFreed();
    return context;
  }

  @Override
  public void free() {
    if (freed) return;
    if (open != null) {
      open.close();
    }
    freed = true;
    sampler.free();
    context.free();
  }

  @Override
  public boolean isFree() {
    return freed;
  }

  @Override
  public void close() {
    free();
  }

  private void checkNotFreed() {
    if (freed) {
      throw new LlamaException("LlamaTranscriber has been freed");
    }
  }

  /**
   * One live audio stream. Each call returns the text transcribed since the previous call;
   * {@link #text()} is the whole transcript.
   */
  public final class Session implements AutoCloseable {

    private final Arena arena = Arena.ofConfined();
    private final ByteArrayOutputStream transcript =
      new ByteArrayOutputStream();
    private MemorySegment stream;
    private MemorySegment batch;
    private int lastToken = TOKEN_NULL;
    private int nPast = 0;
    private int returned = 0;
    private boolean ended = false;

    private Session() {
      MemorySegment params = LlamaRuntime.mtmd_audio_stream_params_default(
        arena
      );
      if (options.stepMs() != null) {
        LlamaRuntime.mtmd_audio_stream_params_set_step_ms(
          params,
          options.stepMs()
        );
      }
      context.getMemory().clear();
      stream = LlamaRuntime.mtmd_audio_stream_init(mtmd.segment, params);
      if (stream == null || stream.address() == 0) {
        arena.close();
        throw new LlamaException("mtmd_audio_stream_init failed");
      }
      batch = LlamaRuntime.llama_batch_ext_init(context.segment);
    }

    /**
     * Pushes audio and transcribes what can be transcribed of it.
     *
     * @param pcm Mono float PCM at {@link #sampleRate()}
     * @return The text transcribed since the previous call
     */
    public String push(float[] pcm) {
      checkOpen();
      if (!ended && pcm.length > 0) {
        try (var local = Arena.ofConfined()) {
          MemorySegment samples = local.allocateFrom(JAVA_FLOAT, pcm);
          if (
            LlamaRuntime.mtmd_audio_stream_push(stream, samples, pcm.length) !=
            0
          ) {
            ended = true;
          }
        }
      }
      return drain();
    }

    /**
     * Transcribes the audio pushed so far without waiting for more, for a pause in the speech. The
     * session stays open.
     *
     * @return The text transcribed since the previous call
     */
    public String pause() {
      checkOpen();
      if (!ended) {
        LlamaRuntime.mtmd_audio_stream_flush(stream, false);
      }
      return drain();
    }

    /**
     * Ends the audio and transcribes the rest of it. Pushing afterwards transcribes nothing.
     *
     * @return The text transcribed since the previous call
     */
    public String finish() {
      checkOpen();
      if (!ended) {
        LlamaRuntime.mtmd_audio_stream_flush(stream, true);
      }
      String text = drain();
      ended = true;
      return text;
    }

    /** The whole transcript so far. */
    public String text() {
      return decodeComplete(transcript.toByteArray());
    }

    /** True once the stream has ended (after {@link #finish()}, or at the model's end of text). */
    public boolean ended() {
      return ended;
    }

    @Override
    public void close() {
      if (stream == null) return;
      LlamaRuntime.mtmd_audio_stream_free(stream);
      LlamaRuntime.llama_batch_ext_free(batch);
      stream = null;
      batch = null;
      arena.close();
      open = null;
    }

    private String drain() {
      while (!ended) {
        int n = LlamaRuntime.mtmd_audio_stream_step(stream, lastToken);
        if (n == -2) {
          throw new LlamaException("the audio stream failed");
        }
        if (n < 0) {
          ended = true;
          break;
        }
        if (n == 0) {
          break;
        }
        if (LlamaRuntime.mtmd_audio_stream_is_seq_start(stream)) {
          context.getMemory().seqRm(0, -1, -1);
          nPast = 0;
        }
        decodeRows(n);
        int token = sampler.sample(context);
        lastToken = token;
        if (vocab.isEog(token)) {
          ended = true;
          break;
        }
        if (!LlamaRuntime.llama_vocab_is_control(vocab.segment, token)) {
          transcript.writeBytes(vocab.tokenToPiece(token));
        }
      }
      String text = text();
      String fresh = text.substring(Math.min(returned, text.length()));
      returned = text.length();
      return fresh;
    }

    private void decodeRows(int n) {
      MemorySegment rows = LlamaRuntime.mtmd_audio_stream_get_output_embd(
        stream
      ).reinterpret((long) n * nEmbd * JAVA_FLOAT.byteSize());
      LlamaRuntime.llama_batch_ext_clear(batch);
      try (var local = Arena.ofConfined()) {
        MemorySegment pos = local.allocate(JAVA_INT, POSITIONS_PER_ROW);
        for (int i = 0; i < n; i++) {
          MemorySegment embd = LlamaRuntime.llama_embd_allocate(local);
          LlamaRuntime.llama_embd_set(
            embd,
            rows.asSlice((long) i * nEmbd * JAVA_FLOAT.byteSize()),
            1,
            nEmbd
          );
          int idx = LlamaRuntime.llama_batch_ext_add_embd(batch, 0, embd);
          if (idx < 0) {
            throw new LlamaException(
              "llama_batch_ext_add_embd failed: %d".formatted(idx)
            );
          }
          pos.setAtIndex(JAVA_INT, 0, nPast);
          pos.setAtIndex(JAVA_INT, 1, nPast);
          pos.setAtIndex(JAVA_INT, 2, nPast);
          pos.setAtIndex(JAVA_INT, 3, 0);
          nPast++;
          if (
            !LlamaRuntime.llama_batch_ext_set_pos(batch, idx, pos) ||
            !LlamaRuntime.llama_batch_ext_set_output_embd(
              batch,
              idx,
              i == n - 1
            )
          ) {
            throw new LlamaException("failed to fill the audio batch");
          }
        }
        int ret = LlamaRuntime.llama_process(
          context.segment,
          LLAMA_PROCESS_TYPE_DECODE,
          batch
        );
        if (ret != 0) {
          throw new LlamaException(
            "llama_process returned non-zero status: %d".formatted(ret)
          );
        }
      }
    }

    private void checkOpen() {
      if (stream == null) {
        throw new LlamaException("the session is closed");
      }
    }
  }

  private static String decodeComplete(byte[] bytes) {
    int end = bytes.length;
    for (int back = 1; back <= Math.min(3, bytes.length); back++) {
      int b = bytes[bytes.length - back] & 0xFF;
      if ((b & 0xC0) == 0x80) {
        continue;
      }
      int need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
      if (need > back) {
        end = bytes.length - back;
      }
      break;
    }
    return new String(bytes, 0, end, StandardCharsets.UTF_8);
  }
}
