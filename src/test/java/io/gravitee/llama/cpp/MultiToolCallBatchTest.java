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

import static io.gravitee.llama.cpp.LlamaCppTest.REASONING_MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.REASONNING_MODEL_TO_DOWNLOAD;
import static io.gravitee.llama.cpp.LlamaCppTest.buildMessages;
import static io.gravitee.llama.cpp.LlamaCppTest.getPrompt;
import static io.gravitee.llama.cpp.LlamaRuntime.ggml_backend_reg_count;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests that the BatchIterator correctly handles multiple tool calls
 * in a single generation — the model should be able to produce several
 * {@code <tool_call>...</tool_call>} blocks without being killed after
 * the first one.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class MultiToolCallBatchTest extends LlamaCppTest {

  private static final Pattern TOOL_CALL_PATTERN = Pattern.compile(
    "<tool_call>\\s*\\{.*?}\\s*</tool_call>",
    Pattern.DOTALL
  );

  /**
   * System prompt that provides two tools and asks the model to call both.
   * The model should produce two separate {@code <tool_call>} blocks.
   */
  private static final String MULTI_TOOL_SYSTEM = """
    You are a helpful assistant that uses tools to answer user questions.
    Only use tools to answer user questions and respect thoroughly the tool instructions.

    # Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {"name": "get_capital", "description": "Get the capital city of a given country.", "parameters": {"type": "object", "properties": {"country": {"type": "string", "description": "The name of the country"}}, "required": ["country"]}}
    {"name": "get_population", "description": "Get the population of a given country.", "parameters": {"type": "object", "properties": {"country": {"type": "string", "description": "The name of the country"}}, "required": ["country"]}}
    </tools>

    For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": "<function-name>", "arguments": {"<arg>": "<value>"}}
    </tool_call>

    You MUST call BOTH tools for EACH country mentioned by the user.
    """;

  private static Arena arena;

  @BeforeAll
  public static void beforeAll() {
    arena = Arena.ofConfined();

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
  void batch_iterator_supports_multiple_tool_calls() {
    var logger = new LlamaLogger(arena);
    logger.setLogging(LlamaLogLevel.INFO);

    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath(
      REASONING_MODEL_PATH,
      REASONNING_MODEL_TO_DOWNLOAD
    );

    var model = new LlamaModel(arena, absolutePath, modelParameters);
    var contextParams = new LlamaContextParams(arena)
      .nCtx(4096)
      .nBatch(4096)
      .nSeqMax(2);
    var context = new LlamaContext(arena, model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(42);

    var prompt = getPrompt(
      model,
      arena,
      buildMessages(
        arena,
        MULTI_TOOL_SYSTEM,
        "What is the capital and population of France?"
      ),
      contextParams
    );

    var state = ConversationState.create(arena, context, tokenizer, sampler, 0)
      .setReasoning("<think>", "</think>")
      .setToolCall("<tool_call>", "</tool_call>")
      .initialize(prompt);

    // Use BatchIterator (the production path) instead of DefaultLlamaIterator
    var batchIterator = new BatchIterator(arena, context).addState(state);

    Map<Integer, StringBuilder> outputs = new HashMap<>();
    outputs.put(0, new StringBuilder());

    while (batchIterator.hasNext()) {
      LlamaOutput output = batchIterator.next();
      outputs.get(output.sequenceId()).append(output.text());
    }

    String fullOutput = outputs.get(0).toString();
    System.out.println("\n=== Multi Tool Call Output ===");
    System.out.println(fullOutput);
    System.out.println("=============================");

    // Count <tool_call>...</tool_call> blocks in the output
    int toolCallCount = 0;
    Matcher matcher = TOOL_CALL_PATTERN.matcher(fullOutput);
    while (matcher.find()) {
      toolCallCount++;
      System.out.println("Tool call " + toolCallCount + ": " + matcher.group());
    }

    System.out.println("Total tool calls: " + toolCallCount);
    System.out.println("Tool tokens: " + state.getToolsTokens());
    System.out.println("Finish reason: " + state.getFinishReason());

    // The model should produce at least 2 tool calls (get_capital + get_population)
    assertThat(toolCallCount).isGreaterThanOrEqualTo(2);

    // Tool tokens should be counted across all tool call blocks
    assertThat(state.getToolsTokens()).isGreaterThan(0);

    // Finish reason must be TOOL_CALL, not STOP
    assertThat(state.getFinishReason()).isEqualTo(FinishReason.TOOL_CALL);

    batchIterator.free();
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
