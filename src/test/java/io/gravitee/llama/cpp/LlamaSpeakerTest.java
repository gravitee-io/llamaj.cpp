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

import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_VL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.VL_MMPROJ;
import static io.gravitee.llama.cpp.LlamaCppTest.VL_MMPROJ_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.VL_TEXT;
import static io.gravitee.llama.cpp.LlamaCppTest.getModelPath;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link LlamaSpeaker}. The speech cases run only when {@code LLAMAJ_TTS_MODEL} and
 * {@code LLAMAJ_TTS_MMPROJ} point at a speech model and its audio-generation projector.
 */
class LlamaSpeakerTest extends LlamaCppTest {

  private static Arena arena;

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);
  }

  @AfterAll
  static void afterAll() {
    LlamaRuntime.llama_backend_free();
    arena.close();
    arena = null;
  }

  @Test
  void wav_is_a_16_bit_mono_riff_file() {
    var speech = new LlamaSpeaker.Speech(
      24000,
      new float[] { 0f, 1f, -1f, 2f },
      1
    );
    var wav = ByteBuffer.wrap(speech.wav()).order(ByteOrder.LITTLE_ENDIAN);
    assertThat(new String(speech.wav(), 0, 4)).isEqualTo("RIFF");
    assertThat(new String(speech.wav(), 8, 4)).isEqualTo("WAVE");
    assertThat(wav.getShort(22)).isEqualTo((short) 1);
    assertThat(wav.getInt(24)).isEqualTo(24000);
    assertThat(wav.getShort(34)).isEqualTo((short) 16);
    assertThat(wav.getInt(40)).isEqualTo(8);
    assertThat(wav.getShort(44)).isZero();
    assertThat(wav.getShort(46)).isEqualTo((short) 32767);
    assertThat(wav.getShort(48)).isEqualTo((short) -32767);
    assertThat(wav.getShort(50)).isEqualTo((short) 32767);
    assertThat(speech.seconds()).isEqualTo(4.0 / 24000);
  }

  @Test
  void a_request_names_a_voice_or_a_reference_not_both() {
    MtmdMedia reference = new MtmdMedia() {
      @Override
      public MemorySegment getMemorySegment() {
        return MemorySegment.NULL;
      }

      @Override
      public void free() {}

      @Override
      public boolean isFree() {
        return true;
      }
    };
    assertThatThrownBy(() ->
      new LlamaSpeaker.Request("x", "a", reference, null, null)
    ).isInstanceOf(IllegalArgumentException.class);
    var request = LlamaSpeaker.Request.of("x")
      .withVoice("a")
      .withSpeakerReference(reference);
    assertThat(request.voice()).isNull();
    assertThat(request.speakerReference()).isSameAs(reference);
    assertThat(request.withVoice("b").speakerReference()).isNull();
  }

  @Test
  void a_vision_projector_does_not_generate_speech() {
    var model = track(
      new LlamaModel(
        arena,
        getModelPath(MODEL_VL_PATH, VL_TEXT),
        new LlamaModelParams(arena)
      )
    );
    var mtmd = track(
      new MtmdContext(
        arena,
        model,
        getModelPath(VL_MMPROJ_PATH, VL_MMPROJ).toAbsolutePath(),
        new MtmdContextParams(arena)
      )
    );
    assertThat(mtmd.capabilities().audioGeneration()).isFalse();
    assertThatThrownBy(() ->
      LlamaSpeaker.create(arena, model, mtmd, LlamaSpeaker.Options.defaults())
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("does not generate speech");
  }

  @Test
  void speaks_a_sentence_whole_and_streamed() {
    var modelPath = System.getenv("LLAMAJ_TTS_MODEL");
    var mmprojPath = System.getenv("LLAMAJ_TTS_MMPROJ");
    assumeTrue(
      modelPath != null &&
        mmprojPath != null &&
        Files.exists(Path.of(modelPath)) &&
        Files.exists(Path.of(mmprojPath)),
      "LLAMAJ_TTS_MODEL / LLAMAJ_TTS_MMPROJ not set"
    );
    var model = track(
      new LlamaModel(arena, Path.of(modelPath), new LlamaModelParams(arena))
    );
    var mtmd = track(
      new MtmdContext(
        arena,
        model,
        Path.of(mmprojPath).toAbsolutePath(),
        new MtmdContextParams(arena)
      )
    );
    assertThat(mtmd.capabilities().audioGeneration()).isTrue();

    try (
      var speaker = LlamaSpeaker.create(
        arena,
        model,
        mtmd,
        LlamaSpeaker.Options.defaults().withSeed(42)
      )
    ) {
      assertThat(speaker.sampleRate()).isPositive();
      var request = LlamaSpeaker.Request.of(
        "Hello, this is a short test of the speech model."
      ).withMaxFrames(200);

      var speech = speaker.speak(request);
      assertThat(speech.frames()).isPositive();
      assertThat(speech.sampleRate()).isEqualTo(speaker.sampleRate());
      assertThat(speech.seconds()).isBetween(0.5, 30.0);
      assertThat(
        IntStream.range(0, speech.pcm().length).allMatch(i ->
          Float.isFinite(speech.pcm()[i])
        )
      ).isTrue();
      assertThat(rms(speech.pcm())).isGreaterThan(0.001);

      List<float[]> chunks = new ArrayList<>();
      var streamed = speaker.speak(request, chunks::add);
      assertThat(chunks).isNotEmpty();
      assertThat(
        chunks
          .stream()
          .mapToInt(c -> c.length)
          .sum()
      ).isEqualTo(streamed.pcm().length);
      assertThat(streamed.seconds()).isBetween(0.5, 30.0);
    }
  }

  private static double rms(float[] pcm) {
    return Math.sqrt(
      IntStream.range(0, pcm.length)
        .mapToDouble(i -> (double) pcm[i] * pcm[i])
        .average()
        .orElse(0)
    );
  }
}
