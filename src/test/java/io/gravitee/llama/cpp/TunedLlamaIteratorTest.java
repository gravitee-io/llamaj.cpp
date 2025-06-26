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
import java.util.List;
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
class TunedLlamaIteratorTest extends LlamaCppTest {

  static final int SEED = new Random().nextInt();

  static final String ENGLISH_GRAMMAR =
    """
            root        ::= en-char+ ([ \\t\\n] en-char+)*
            en-char     ::= letter | digit | punctuation
            letter      ::= [a-zA-Z]
            digit       ::= [0-9]
            punctuation ::= [!"#$%&'()*+,-./:;<=>?@[\\\\\\]^_`{|}~]
            """;

  static Stream<Arguments> params_that_allow_llama_generation() {
    return Stream.of(
      Arguments.of(SYSTEM, "What is the capital of France?", false),
      Arguments.of(SYSTEM, "What is the capital of the UK?", false),
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
  void llama_tuned_generation(String system, String input, boolean allowLoraAdapter) {
    int inputToken = -1;
    int outputToken = -1;
    var logger = new LlamaLogger(arena);
    logger.setLogging(LlamaLogLevel.DEBUG);

    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath();

    var model = new LlamaModel(arena, absolutePath, modelParameters);
    if (allowLoraAdapter) {
      model.initLoraAdapter(arena, getLoraAdapterPath());
    }

    var contextParams = new LlamaContextParams(arena)
      .nCtx(512)
      .nBatch(512)
      .attentionType(AttentionType.CAUSAL)
      .embeddings(false)
      .offloadKQV(false)
      .flashAttn(false)
      .noPerf(false);

    var vocab = new LlamaVocab(model);
    var sampler = new LlamaSampler(arena)
      .seed(SEED)
      .temperature(0.75f)
      .topK(40)
      .topP(0.2f, 40)
      .minP(0.05f, 40)
      .mirostat(SEED, 3, 0.1f)
      .grammar(vocab, ENGLISH_GRAMMAR, "root")
      .penalties(10, 1.2f, 0.3f, 0.0f);

    var prompt = getPrompt(model, arena, buildMessages(arena, system, input), contextParams);

    var context = new LlamaContext(model, contextParams);
    var tokenizer = new LlamaTokenizer(vocab, context);

    var it = new SimpleLlamaIterator(arena, context, tokenizer, sampler)
      .setStopStrings(List.of("."))
      .setQuota(10)
      .initialize(prompt);

    String output = it.stream().map(LlamaOutput::content).reduce((a, b) -> a + b).orElse("");

    inputToken = it.getInputTokens();
    outputToken = it.getOutputTokens();

    assertThat(inputToken).isGreaterThan(0);
    assertThat(outputToken).isGreaterThan(0);
    assertThat(output).isNotNull();

    System.out.println(output);

    context.free();
    sampler.free();
    model.free();

    LlamaRuntime.llama_backend_free();
  }

  @AfterAll
  public static void afterAll() {
    arena = null;
    LlamaRuntime.llama_backend_free();
  }
}
