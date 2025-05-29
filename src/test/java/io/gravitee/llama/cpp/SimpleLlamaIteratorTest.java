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

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

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
      Arguments.of(SYSTEM, "What is the capital of France?", "Paris"),
      Arguments.of(SYSTEM, "What is the capital of England?", "London"),
      Arguments.of(SYSTEM, "What is the capital of Poland?", "Warsaw")
    );
  }

  private static Arena arena;

  @BeforeAll
  public static void beforeAll() {
    LlamaCppTest.beforeAll();
    LlamaRuntime.ggml_backend_load_all();
    arena = Arena.ofConfined();
  }

  @ParameterizedTest
  @MethodSource("params_that_allow_llama_generation")
  void llama_simple_generation(String system, String input, String expected) {
    String output = "";
    int inputToken = -1;
    int outputToken = -1;
    var logger = new LlamaLogger(arena);
    logger.setLogging(LlamaLogLevel.ERROR);

    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath();

    var model = new LlamaModel(arena, absolutePath, modelParameters);

    var contextParams = new LlamaContextParams(arena);

    var vocab = new LlamaVocab(model);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());
    var prompt = getPrompt(model, arena, buildMessages(arena, system, input), contextParams);

    var it = new SimpleLlamaIterator(arena, model, contextParams, vocab, sampler).initialize(prompt);
    while (it.hasNext()) {
      output += it.next().content();
    }

    it.close();
    inputToken = it.getInputTokens();
    outputToken = it.getInputTokens();

    assertThat(inputToken).isGreaterThan(0);
    assertThat(outputToken).isGreaterThan(0);
    assertThat(output).containsIgnoringCase(expected);
    System.out.println(output);

    sampler.free();
    model.free();
  }

  @AfterAll
  public static void afterAll() {
    arena = null;
  }
}
