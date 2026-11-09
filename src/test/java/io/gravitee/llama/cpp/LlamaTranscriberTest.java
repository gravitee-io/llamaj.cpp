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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link LlamaTranscriber}. The transcription cases run only when
 * {@code LLAMAJ_ASR_MODEL} and {@code LLAMAJ_ASR_MMPROJ} point at a streaming speech-to-text model
 * and its projector; the round trip also needs {@code LLAMAJ_TTS_MODEL} and
 * {@code LLAMAJ_TTS_MMPROJ}.
 */
class LlamaTranscriberTest extends LlamaCppTest {

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
  void a_vision_model_does_not_transcribe_streams() {
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
    assertThat(mtmd.capabilities().audioStream()).isFalse();
    assertThatThrownBy(() ->
      LlamaTranscriber.create(
        arena,
        model,
        mtmd,
        LlamaTranscriber.Options.defaults()
      )
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("does not transcribe");
  }

  @Test
  void transcribes_a_recording_pushed_as_a_live_stream() throws Exception {
    var asr = model("LLAMAJ_ASR_MODEL", "LLAMAJ_ASR_MMPROJ");
    try (
      var transcriber = LlamaTranscriber.create(
        arena,
        asr.model(),
        asr.mtmd(),
        LlamaTranscriber.Options.defaults()
      )
    ) {
      float[] pcm = AudioLoader.loadAudio(
        Files.readAllBytes(
          Path.of(getClass().getClassLoader().getResource("audio.wav").toURI())
        ),
        transcriber.sampleRate()
      );

      var pieces = new StringBuilder();
      String text;
      try (var session = transcriber.open()) {
        int step = transcriber.sampleRate() / 5;
        for (int from = 0; from < pcm.length; from += step) {
          pieces.append(
            session.push(
              Arrays.copyOfRange(pcm, from, Math.min(pcm.length, from + step))
            )
          );
        }
        pieces.append(session.finish());
        assertThat(session.ended()).isTrue();
        assertThat(session.push(new float[] { 0f })).isEmpty();
        text = session.text();
        assertThatThrownBy(transcriber::open).isInstanceOf(
          LlamaException.class
        );
      }
      assertThat(text).isNotBlank();
      assertThat(pieces.toString()).isEqualTo(text);
      assertThat(transcriber.transcribe(pcm)).isEqualTo(text);
    }
  }

  @Test
  void transcribes_speech_from_the_speaker() throws Exception {
    var asr = model("LLAMAJ_ASR_MODEL", "LLAMAJ_ASR_MMPROJ");
    var tts = model("LLAMAJ_TTS_MODEL", "LLAMAJ_TTS_MMPROJ");
    LlamaSpeaker.Speech speech;
    try (
      var speaker = LlamaSpeaker.create(
        arena,
        tts.model(),
        tts.mtmd(),
        LlamaSpeaker.Options.defaults().withSeed(42)
      )
    ) {
      speech = speaker.speak(
        LlamaSpeaker.Request.of("The weather in Paris is sunny today.")
      );
    }
    try (
      var transcriber = LlamaTranscriber.create(
        arena,
        asr.model(),
        asr.mtmd(),
        LlamaTranscriber.Options.defaults()
      )
    ) {
      float[] pcm = AudioLoader.loadAudio(
        speech.wav(),
        transcriber.sampleRate()
      );
      String text = transcriber.transcribe(pcm).toLowerCase();
      assertThat(text).contains("weather").contains("paris").contains("sunny");
    }
  }

  private record Loaded(LlamaModel model, MtmdContext mtmd) {}

  private Loaded model(String modelVar, String mmprojVar) {
    var modelPath = System.getenv(modelVar);
    var mmprojPath = System.getenv(mmprojVar);
    assumeTrue(
      modelPath != null &&
        mmprojPath != null &&
        Files.exists(Path.of(modelPath)) &&
        Files.exists(Path.of(mmprojPath)),
      modelVar + " / " + mmprojVar + " not set"
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
    return new Loaded(model, mtmd);
  }
}
