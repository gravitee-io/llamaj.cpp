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

import static io.gravitee.llama.cpp.LlamaRuntime.ggml_backend_reg_count;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests for Qwen3-VL-2B-Instruct-GGUF model.
 */
class MultimodalIteratorTest extends LlamaCppTest {

  private static Arena arena = Arena.ofShared();

  @BeforeAll
  public static void beforeAll() {
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

    System.out.println("****************************");
    System.out.println("Libraries loaded at: " + libPath);
    System.out.println(
      "Number of devices registered: " + ggml_backend_reg_count()
    );
    System.out.println("****************************");
  }

  @Test
  void must_test_with_default_iterator()
    throws IOException, URISyntaxException {
    int inputToken = -1;
    int outputToken = -1;
    var logger = new LlamaLogger(arena);
    logger.setLogging(LlamaLogLevel.DEBUG);

    // 1. Load the main Llama model
    var modelParameters = new LlamaModelParams(arena);
    Path mainModelAbsolutePath = getModelPath(MODEL_VL_PATH, VL_TEXT);
    // Add existence checks for main model
    if (
      !Files.exists(mainModelAbsolutePath) ||
      !Files.isRegularFile(mainModelAbsolutePath)
    ) {
      System.err.println(
        "Main model file does not exist or is not a regular file: " +
          mainModelAbsolutePath
      );
      // Optionally, throw an exception or skip the test if the model is crucial
      throw new IOException(
        "Main model file not found: " + mainModelAbsolutePath
      );
    }
    System.out.println("Main model file exists: " + mainModelAbsolutePath);
    var llamaModel = new LlamaModel(
      arena,
      mainModelAbsolutePath,
      modelParameters
    );

    // 2. Setup MtmdContextParams and MtmdContext for multi-modal processing
    var mtmdContextParams = new MtmdContextParams(arena)
      .useGpu(true) // Assuming GPU is available and desired for multi-modal
      .mediaMarker("<IMG>") // As observed from MtmdContextTest
      .printTimings(true);

    Path mmprojModelAbsolutePath = getModelPath(VL_MMPROJ_PATH, VL_MMPROJ);
    // Add existence checks for mmproj model
    if (
      !Files.exists(mmprojModelAbsolutePath) ||
      !Files.isRegularFile(mmprojModelAbsolutePath)
    ) {
      System.err.println(
        "MMProj model file does not exist or is not a regular file: " +
          mmprojModelAbsolutePath
      );
      throw new IOException(
        "MMProj model file not found: " + mmprojModelAbsolutePath
      );
    }
    System.out.println("MMProj model file exists: " + mmprojModelAbsolutePath);
    var mtmdContext = new MtmdContext(
      arena,
      llamaModel,
      mmprojModelAbsolutePath.toAbsolutePath(),
      mtmdContextParams
    );

    // 3. Create a regular LlamaContext for model inference
    var llamaContextParams = new LlamaContextParams(arena).noPerf(false);
    var llamaContext = new LlamaContext(arena, llamaModel, llamaContextParams);

    // 4. Load the image
    Path imagePath = Path.of(
      getClass().getClassLoader().getResource("dog.jpg").toURI()
    );
    var mtmdImage = MtmdImage.fromFile(arena, imagePath);

    // 5. Prepare ConversationState
    var vocab = new LlamaVocab(llamaModel);
    var tokenizer = new LlamaTokenizer(vocab, llamaContext);
    var sampler = new LlamaSampler(arena).greedy().seed(new Random().nextInt());

    String promptText = "USER: What is in this image?\n<IMG>\nASSISTANT:";
    var state = ConversationState.create(
      arena,
      llamaContext,
      tokenizer,
      sampler
    )
      .initialize(promptText)
      .setImages(List.of(mtmdImage));

    // 6. Instantiate DefaultLlamaIterator with MtmdContext
    var it = new DefaultLlamaIterator(state, mtmdContext);

    // 7. Perform generation and assertions
    String output = it
      .stream()
      .reduce(LlamaOutput::merge)
      .orElse(new LlamaOutput("", 0))
      .content();
    System.out.println(output);

    inputToken = state.getInputTokens();
    outputToken = state.getAnswerTokens();

    assertThat(inputToken).isGreaterThan(0);
    assertThat(outputToken).isGreaterThan(0);
    assertThat(state.getFinishReason()).isIn(
      FinishReason.EOS,
      FinishReason.LENGTH,
      FinishReason.STOP
    );

    // Verify performance metrics are extracted correctly
    LlamaPerformance perf = it.getPerformance();
    assertThat(perf).isNotNull();
    assertThat(perf.context()).isNotNull();
    assertThat(perf.sampler()).isNotNull();

    System.out.println("=== Performance Debug ===");
    System.out.printf("Start time: %.4f ms%n", perf.context().startTimeMs());
    System.out.printf("Load time: %.4f ms%n", perf.context().loadTimeMs());
    System.out.printf(
      "Prompt eval time: %.4f ms%n",
      perf.context().promptEvalTimeMs()
    );
    System.out.printf("Eval time: %.4f ms%n", perf.context().evalTimeMs());
    System.out.printf(
      "Prompt tokens evaluated: %d%n",
      perf.context().promptTokensEvaluated()
    );
    System.out.printf(
      "Tokens generated: %d%n",
      perf.context().tokensGenerated()
    );
    assertThat(perf.context().tokensReused())
      .as("Tokens reused should be >= 0")
      .isGreaterThanOrEqualTo(0);
    System.out.printf("Tokens reused: %d%n", perf.context().tokensReused());
    System.out.printf(
      "Sampling time: %.4f ms%n",
      perf.sampler().samplingTimeMs()
    );
    System.out.printf("Sample count: %d%n", perf.sampler().sampleCount());
    System.out.println("========================");

    // Verify context metrics
    assertThat(perf.context().promptTokensEvaluated())
      .as("Prompt tokens should be evaluated")
      .isGreaterThan(0);
    assertThat(perf.context().tokensGenerated())
      .as("Tokens should be generated")
      .isGreaterThan(0);
    assertThat(perf.context().evalTimeMs())
      .as("Generation time must be non-negative if tokens were generated")
      .isGreaterThanOrEqualTo(0.0);

    // Verify speed calculations
    assertThat(perf.generationTokensPerSecond())
      .as("Generation speed should be non-negative")
      .isGreaterThanOrEqualTo(0.0);

    // Verify sampler metrics
    assertThat(perf.sampler().sampleCount())
      .as("Samples should be taken")
      .isGreaterThan(0);

    System.out.printf(
      "Performance: %.2f tokens/sec (prompt: %.2f tokens/sec)%n",
      perf.generationTokensPerSecond(),
      perf.promptTokensPerSecond()
    );

    // 8. Free resources
    mtmdImage.free(); // Free the image
    mtmdContext.free(); // Free the multi-modal context
    llamaContext.free();
    sampler.free();
    llamaModel.free();
  }

  @Test
  void must_test_audio_with_default_iterator()
    throws IOException, URISyntaxException, javax.sound.sampled.UnsupportedAudioFileException {
    int inputToken = -1;
    int outputToken = -1;
    var logger = new LlamaLogger(arena);
    logger.setLogging(LlamaLogLevel.DEBUG);

    // 1. Load the Ultravox audio model
    var modelParameters = new LlamaModelParams(arena);
    Path mainModelAbsolutePath = getModelPath(
      AUDIO_MODEL_PATH,
      AUDIO_MODEL_TO_DOWNLOAD
    );
    if (
      !Files.exists(mainModelAbsolutePath) ||
      !Files.isRegularFile(mainModelAbsolutePath)
    ) {
      System.err.println(
        "Audio model file does not exist or is not a regular file: " +
          mainModelAbsolutePath
      );
      throw new IOException(
        "Audio model file not found: " + mainModelAbsolutePath
      );
    }
    System.out.println("Audio model file exists: " + mainModelAbsolutePath);
    var llamaModel = new LlamaModel(
      arena,
      mainModelAbsolutePath,
      modelParameters
    );

    // 2. Setup MtmdContextParams and MtmdContext for audio processing
    var mtmdContextParams = new MtmdContextParams(arena)
      .useGpu(true)
      .printTimings(true);

    Path mmprojModelAbsolutePath = getModelPath(
      AUDIO_MMPROJ_PATH,
      AUDIO_MMPROJ_TO_DOWNLOAD
    );
    if (
      !Files.exists(mmprojModelAbsolutePath) ||
      !Files.isRegularFile(mmprojModelAbsolutePath)
    ) {
      System.err.println(
        "Audio mmproj file does not exist or is not a regular file: " +
          mmprojModelAbsolutePath
      );
      throw new IOException(
        "Audio mmproj file not found: " + mmprojModelAbsolutePath
      );
    }
    System.out.println("Audio mmproj file exists: " + mmprojModelAbsolutePath);
    var mtmdContext = new MtmdContext(
      arena,
      llamaModel,
      mmprojModelAbsolutePath.toAbsolutePath(),
      mtmdContextParams
    );

    // 3. Create a regular LlamaContext for model inference
    var llamaContextParams = new LlamaContextParams(arena).noPerf(false);
    var llamaContext = new LlamaContext(arena, llamaModel, llamaContextParams);

    // 4. Load audio file
    Path audioPath = Path.of(
      getClass().getClassLoader().getResource("test-2.wav").toURI()
    );
    var mtmdAudio = MtmdAudio.fromFile(
      arena,
      audioPath,
      mtmdContext.getAudioBitrate()
    );
    System.out.println(
      "Audio file loaded successfully with bitrate: " +
        mtmdContext.getAudioBitrate()
    );

    // 5. Prepare ConversationState with audio using Llama 3.2 chat template
    var vocab = new LlamaVocab(llamaModel);
    var tokenizer = new LlamaTokenizer(vocab, llamaContext);
    var sampler = new LlamaSampler(arena).greedy().seed(new Random().nextInt());

    // Use the model's chat template with the default media marker
    String mediaMarker = mtmdContextParams.mediaMarker();
    var messages = new LlamaChatMessages(
      arena,
      List.of(
        new LlamaChatMessage(
          arena,
          Role.USER,
          mediaMarker + "\nWhat is happening in this audio?"
        )
      )
    );
    String promptText = getPrompt(
      llamaModel,
      arena,
      messages,
      llamaContextParams
    );
    System.out.println("Formatted prompt: " + promptText);

    var state = ConversationState.create(
      arena,
      llamaContext,
      tokenizer,
      sampler
    )
      .initialize(promptText)
      .setMedia(List.of(mtmdAudio));

    // 6. Instantiate DefaultLlamaIterator with MtmdContext and audio
    var it = new DefaultLlamaIterator(state, mtmdContext);

    // 7. Perform generation and assertions
    String output = it
      .stream()
      .reduce(LlamaOutput::merge)
      .orElse(new LlamaOutput("", 0))
      .content();
    System.out.println("=== Audio Processing Output ===");
    System.out.println(output);
    System.out.println("================================");

    inputToken = state.getInputTokens();
    outputToken = state.getAnswerTokens();

    assertThat(inputToken).isGreaterThan(0);
    assertThat(outputToken).isGreaterThan(0);
    assertThat(state.getFinishReason()).isIn(
      FinishReason.EOS,
      FinishReason.LENGTH,
      FinishReason.STOP
    );

    // Verify performance metrics are extracted correctly
    LlamaPerformance perf = it.getPerformance();
    assertThat(perf).isNotNull();
    assertThat(perf.context()).isNotNull();
    assertThat(perf.sampler()).isNotNull();

    System.out.println("=== Audio Performance Debug ===");
    System.out.printf("Start time: %.4f ms%n", perf.context().startTimeMs());
    System.out.printf("Load time: %.4f ms%n", perf.context().loadTimeMs());
    System.out.printf(
      "Prompt eval time: %.4f ms%n",
      perf.context().promptEvalTimeMs()
    );
    System.out.printf("Eval time: %.4f ms%n", perf.context().evalTimeMs());
    System.out.printf(
      "Prompt tokens evaluated: %d%n",
      perf.context().promptTokensEvaluated()
    );
    System.out.printf(
      "Tokens generated: %d%n",
      perf.context().tokensGenerated()
    );
    assertThat(perf.context().tokensReused())
      .as("Tokens reused should be >= 0")
      .isGreaterThanOrEqualTo(0);
    System.out.printf("Tokens reused: %d%n", perf.context().tokensReused());
    System.out.printf(
      "Sampling time: %.4f ms%n",
      perf.sampler().samplingTimeMs()
    );
    System.out.printf("Sample count: %d%n", perf.sampler().sampleCount());
    System.out.println("===============================");

    // Verify context metrics
    assertThat(perf.context().promptTokensEvaluated())
      .as("Prompt tokens should be evaluated")
      .isGreaterThan(0);
    assertThat(perf.context().tokensGenerated())
      .as("Tokens should be generated")
      .isGreaterThan(0);
    assertThat(perf.context().evalTimeMs())
      .as("Generation time must be non-negative if tokens were generated")
      .isGreaterThanOrEqualTo(0.0);

    // Verify speed calculations
    assertThat(perf.generationTokensPerSecond())
      .as("Generation speed should be non-negative")
      .isGreaterThanOrEqualTo(0.0);

    // Verify sampler metrics
    assertThat(perf.sampler().sampleCount())
      .as("Samples should be taken")
      .isGreaterThan(0);

    System.out.printf(
      "Audio Performance: %.2f tokens/sec (prompt: %.2f tokens/sec)%n",
      perf.generationTokensPerSecond(),
      perf.promptTokensPerSecond()
    );

    // 8. Free resources
    mtmdAudio.free(); // Free the audio
    mtmdContext.free(); // Free the multi-modal context
    llamaContext.free();
    sampler.free();
    llamaModel.free();
  }
}
