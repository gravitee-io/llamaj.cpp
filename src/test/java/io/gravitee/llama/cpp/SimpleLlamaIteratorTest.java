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
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Random;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class SimpleLlamaIteratorTest extends LlamaCppTest {

  static Stream<Arguments> params_that_allow_llama_generation() {
    return Stream.of(
      Arguments.of(SYSTEM, "What is the capital of France?", false),
      Arguments.of(SYSTEM, "What is the capital of England?", false),
      Arguments.of(SYSTEM, "What is the capital of Poland?", false),
      Arguments.of(SYSTEM, "What is the capital of France?", true)
    );
  }

  private static Arena arena;

  @BeforeAll
  public static void beforeAll() {
    arena = Arena.ofConfined();

    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

    System.out.println("****************************");
    System.out.println("Libraries loaded at: " + libPath);
    System.out.println("Number of devices registered: " + ggml_backend_reg_count());
    System.out.println("****************************");
  }

  @ParameterizedTest
  @MethodSource("params_that_allow_llama_generation")
  void llama_simple_generation(String system, String input, boolean allowLoraAdapter) {
    int inputToken = -1;
    int outputToken = -1;
    var logger = new LlamaLogger(arena);
    logger.setLogging(LlamaLogLevel.DEBUG);

    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);

    var model = new LlamaModel(arena, absolutePath, modelParameters);
    if (allowLoraAdapter) {
      model.initLoraAdapter(arena, getModelPath(LORA_ADATAPTER_PATH, LORA_ADAPTER_TO_DOWNLOAD));
    }

    var contextParams = new LlamaContextParams(arena).noPerf(false);
    var context = new LlamaContext(model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());
    var prompt = getPrompt(model, arena, buildMessages(arena, system, input), contextParams);

    var it = new DefaultLlamaIterator(arena, context, tokenizer, sampler).initialize(prompt);

    String output = it.stream().reduce(LlamaOutput::merge).orElse(new LlamaOutput("", 0)).content();
    System.out.println(output);

    inputToken = it.getInputTokens();
    outputToken = it.getAnswerTokens();

    assertThat(inputToken).isGreaterThan(0);
    assertThat(outputToken).isGreaterThan(0);
    assertThat(it.getFinishReason()).isIn(FinishReason.EOS, FinishReason.LENGTH, FinishReason.STOP);

    // Verify performance metrics are extracted correctly
    LlamaPerformance perf = it.getPerformance();
    assertThat(perf).isNotNull();
    assertThat(perf.context()).isNotNull();
    assertThat(perf.sampler()).isNotNull();

    System.out.println("=== Performance Debug ===");
    System.out.printf("Start time: %.4f ms%n", perf.context().startTimeMs());
    System.out.printf("Load time: %.4f ms%n", perf.context().loadTimeMs());
    System.out.printf("Prompt eval time: %.4f ms%n", perf.context().promptEvalTimeMs());
    System.out.printf("Eval time: %.4f ms%n", perf.context().evalTimeMs());
    System.out.printf("Prompt tokens evaluated: %d%n", perf.context().promptTokensEvaluated());
    System.out.printf("Tokens generated: %d%n", perf.context().tokensGenerated());
    System.out.printf("Tokens reused: %d%n", perf.context().tokensReused());
    System.out.printf("Sampling time: %.4f ms%n", perf.sampler().samplingTimeMs());
    System.out.printf("Sample count: %d%n", perf.sampler().sampleCount());
    System.out.println("========================");

    // Verify context metrics
    assertThat(perf.context().promptTokensEvaluated()).as("Prompt tokens should be evaluated").isGreaterThan(0);
    assertThat(perf.context().tokensGenerated()).as("Tokens should be generated").isGreaterThan(0);
    assertThat(perf.context().evalTimeMs())
      .as("Generation time must be positive if tokens were generated")
      .isGreaterThan(0.0);

    // Verify speed calculations
    assertThat(perf.generationTokensPerSecond()).as("Generation speed should be positive").isGreaterThan(0.0);

    // Verify sampler metrics
    assertThat(perf.sampler().sampleCount()).as("Samples should be taken").isGreaterThan(0);

    System.out.printf(
      "Performance: %.2f tokens/sec (prompt: %.2f tokens/sec)%n",
      perf.generationTokensPerSecond(),
      perf.promptTokensPerSecond()
    );

    context.free();
    sampler.free();
    model.free();
  }

  @AfterAll
  public static void afterAll() {
    arena = null;
    LlamaRuntime.llama_backend_free();
  }
}
