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
class ToolCallLlamaIteratorTest extends LlamaCppTest {

  private static final String TOOL_CALL_SYSTEM =
    """
          You are a helpful assistant that uses tools to answer user questions.
          
          # Tools
          
          You may call one or more functions to assist with the user query.
          
          You are provided with function signatures within <tools></tools> XML tags:
          <tools>
          {"name": "get_capital", "description": "Get the capital city of a given country.", "parameters": {"type": "object", "properties": {"country": {"type": "string", "description": "The name of the country"}}, "required": ["country"]}}
          </tools>
          
          For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
          <tool_call>
          {"name": "get_capital", "arguments": {"country": "<country-name>"}}
          </tool_call>
          """;

  static Stream<Arguments> params_that_allow_llama_generation() {
    return Stream.of(
      Arguments.of(TOOL_CALL_SYSTEM, "What is the capital of England?"),
      Arguments.of(TOOL_CALL_SYSTEM, "What is the capital of Poland?"),
      Arguments.of(TOOL_CALL_SYSTEM, "What is the capital of France?")
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
  void llama_simple_generation(String system, String input) {
    var logger = new LlamaLogger(arena);
    logger.setLogging(LlamaLogLevel.DEBUG);

    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath(REASONING_MODEL_PATH, REASONNING_MODEL_TO_DOWNLOAD);

    var model = new LlamaModel(arena, absolutePath, modelParameters);

    var contextParams = new LlamaContextParams(arena);
    var context = new LlamaContext(model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());
    var prompt = getPrompt(model, arena, buildMessages(arena, system, input), contextParams);

    var it = new DefaultLlamaIterator(arena, context, tokenizer, sampler)
      .setReasoning("<think>", "</think>")
      .setToolCall("<tool_call>", "</tool_call>")
      .initialize(prompt);

    LlamaOutput output = it.stream().reduce(LlamaOutput::merge).orElse(new LlamaOutput("", 0));
    System.out.println(output);

    int inputTokens = it.getInputTokens();
    int answerTokens = it.getAnswerTokens();
    int reasoningTokens = it.getReasoningTokens();
    int toolCallTokens = it.getToolCallTokens();

    assertThat(inputTokens).isGreaterThan(0);
    assertThat(answerTokens).isGreaterThan(0);
    assertThat(reasoningTokens).isGreaterThan(0);
    assertThat(toolCallTokens).isGreaterThan(0);

    int outputTokens = answerTokens + reasoningTokens + toolCallTokens;

    assertThat(output.numberOfTokens()).isEqualTo(outputTokens);
    assertThat(it.getTotalTokenCount()).isEqualTo(inputTokens + outputTokens);

    assertThat(it.getFinishReason()).isIn(FinishReason.TOOL_CALL);

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
