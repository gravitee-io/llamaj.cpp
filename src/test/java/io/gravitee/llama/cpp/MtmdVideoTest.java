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
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MtmdVideoTest extends LlamaCppTest {

  private static final String IM_START = "<|" + "im_start|>";
  private static final String IM_END = "<|" + "im_end|>";

  private static Arena arena;

  @TempDir
  static Path tmp;

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
  void a_vision_model_reads_a_video_as_frames() throws Exception {
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
        new MtmdContextParams(arena).useGpu(true)
      )
    );
    var capabilities = mtmd.capabilities();
    assertThat(capabilities.vision()).isTrue();
    assertThat(capabilities.chat()).isTrue();
    assertThat(capabilities.audioGeneration()).isFalse();
    assumeTrue(capabilities.video(), "llama.cpp built without video support");
    Path clip = clipOfTheTestImage();

    var video = MtmdVideo.fromFile(
      arena,
      mtmd,
      clip,
      MtmdVideo.Options.defaults().withFpsTarget(2)
    );
    String prefill = "Answer:";
    try (
      var decider = LabelDecider.custom(
        arena,
        model,
        mtmd,
        LlamaDecider.Options.defaults(),
        prefill,
        List.of(" yes", " no")
      )
    ) {
      assertThat(video.info().width()).isPositive();
      assertThat(video.info().height()).isPositive();
      assertThat(video.info().fps()).isEqualTo(2f);

      String prompt = String.join(
        "\n",
        IM_START + "system\nAnswer yes or no." + IM_END,
        IM_START +
          "user\n" +
          mtmd.marker() +
          "\nIs the person in the video wearing glasses?" +
          IM_END,
        IM_START + "assistant\n" + prefill
      );
      float[] scores = decider.score(MediaPrompt.of(prompt, List.of(video), 2));
      assertThat(scores).hasSize(2);
      assertThat(scores[0]).isGreaterThan(scores[1]);
    } finally {
      video.free();
    }
    assertThat(video.isFree()).isTrue();

    byte[] clipBytes = Files.readAllBytes(clip);
    assertThatThrownBy(() -> mtmd.bitmapInitFromBuf(arena, clipBytes))
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("MtmdVideo");
    byte[] imageBytes = Files.readAllBytes(testImage());
    assertThatThrownBy(() ->
      MtmdVideo.fromBytes(arena, mtmd, imageBytes, MtmdVideo.Options.defaults())
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("not a video");
  }

  @Test
  void the_default_handlers_pick_video_files_by_extension() {
    var handler = new VideoHandler();
    assertThat(handler.supports(Path.of("clip.MP4"))).isTrue();
    assertThat(handler.supports(Path.of("clip.webm"))).isTrue();
    assertThat(handler.supports(Path.of("photo.jpg"))).isFalse();
    assertThat(handler.supports(new byte[] { 1 })).isFalse();
  }

  private Path testImage() throws Exception {
    return Path.of(getClass().getClassLoader().getResource("man.jpg").toURI());
  }

  private Path clipOfTheTestImage() throws Exception {
    Path clip = tmp.resolve("clip.mp4");
    Process process;
    try {
      process = new ProcessBuilder(
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-i",
        testImage().toString(),
        "-t",
        "2",
        "-r",
        "4",
        "-vf",
        "scale=192:192",
        "-pix_fmt",
        "yuv420p",
        clip.toString()
      )
        .redirectErrorStream(true)
        .start();
    } catch (java.io.IOException e) {
      assumeTrue(false, "ffmpeg not found");
      throw e;
    }
    boolean done = process.waitFor(60, TimeUnit.SECONDS);
    assumeTrue(
      done && process.exitValue() == 0 && Files.size(clip) > 0,
      "ffmpeg could not encode the clip"
    );
    return clip;
  }
}
