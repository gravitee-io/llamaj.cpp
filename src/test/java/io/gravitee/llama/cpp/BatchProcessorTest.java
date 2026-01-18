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
import static org.assertj.core.api.AssertionsForClassTypes.assertThatThrownBy;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests parallel processing of multiple conversations in a single batch.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class BatchProcessorTest extends LlamaCppTest {

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

  @Test
  void parallel_conversations_generate_tokens() {
    var logger = new LlamaLogger(arena);
    logger.setLogging(LlamaLogLevel.INFO);

    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);

    var model = new LlamaModel(arena, absolutePath, modelParameters);

    // Use a larger context to support multiple sequences
    var contextParams = new LlamaContextParams(arena)
      .nCtx(2048)
      .nBatch(512)
      .nSeqMax(4) // Support up to 4 parallel sequences
      .noPerf(false);
    var context = new LlamaContext(model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());

    // Create 3 different conversations
    String[] prompts = {
      getPrompt(model, arena, buildMessages(arena, SYSTEM, "What is the capital of France?"), contextParams),
      getPrompt(model, arena, buildMessages(arena, SYSTEM, "What is the capital of England?"), contextParams),
      getPrompt(model, arena, buildMessages(arena, SYSTEM, "What is the capital of Poland?"), contextParams),
    };

    // Create conversation states with different sequence IDs (0, 1, 2)
    var state1 = ConversationState.create(arena, context, tokenizer, sampler, 0).setMaxTokens(50).initialize(prompts[0]);
    var state2 = ConversationState.create(arena, context, tokenizer, sampler, 1).setMaxTokens(50).initialize(prompts[1]);
    var state3 = ConversationState.create(arena, context, tokenizer, sampler, 2).setMaxTokens(50).initialize(prompts[2]);

    // Create parallel iterator - prompts are auto-processed when states are added
    var parallel = new BatchIterator(arena, context).addState(state1).addState(state2).addState(state3);

    // Map to accumulate outputs per sequence
    Map<Integer, StringBuilder> sequenceOutputs = new HashMap<>();
    sequenceOutputs.put(0, new StringBuilder());
    sequenceOutputs.put(1, new StringBuilder());
    sequenceOutputs.put(2, new StringBuilder());

    // Generate tokens in parallel using stream
    long startTime = System.nanoTime();

    parallel.stream().forEach(output -> sequenceOutputs.get(output.sequenceId()).append(output.text()));

    long endTime = System.nanoTime();
    double durationMs = (endTime - startTime) / 1_000_000.0;

    System.out.println("\n=== Parallel Generation Results ===");
    System.out.println("Conversation 1 (seq 0): " + sequenceOutputs.get(0).toString());
    System.out.println("  Tokens: " + state1.getAnswerTokens());
    System.out.println("Conversation 2 (seq 1): " + sequenceOutputs.get(1).toString());
    System.out.println("  Tokens: " + state2.getAnswerTokens());
    System.out.println("Conversation 3 (seq 2): " + sequenceOutputs.get(2).toString());
    System.out.println("  Tokens: " + state3.getAnswerTokens());
    System.out.println("Total time: " + durationMs + " ms");
    System.out.println("===================================");

    // Verify all conversations generated tokens
    assertThat(state1.getAnswerTokens()).isGreaterThan(0);
    assertThat(state2.getAnswerTokens()).isGreaterThan(0);
    assertThat(state3.getAnswerTokens()).isGreaterThan(0);

    // Verify all conversations have finish reasons
    assertThat(state1.getFinishReason()).isNotNull();
    assertThat(state2.getFinishReason()).isNotNull();
    assertThat(state3.getFinishReason()).isNotNull();

    parallel.free();
    context.free();
    sampler.free();
    model.free();
  }

  @Test
  void test_batch_iterator_with_next_pattern() {
    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = new LlamaModel(arena, absolutePath, modelParameters);
    var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512).nSeqMax(4);
    var context = new LlamaContext(model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());

    String[] prompts = {
      getPrompt(model, arena, buildMessages(arena, SYSTEM, "Count from 1 to 100"), contextParams),
      getPrompt(model, arena, buildMessages(arena, SYSTEM, "Count from 200 to 300"), contextParams),
    };

    var state1 = ConversationState.create(arena, context, tokenizer, sampler, 0).setMaxTokens(50).initialize(prompts[0]);
    var state2 = ConversationState.create(arena, context, tokenizer, sampler, 1).setMaxTokens(50).initialize(prompts[1]);

    var parallel = new BatchIterator(arena, context).addState(state1).addState(state2);

    // Test removeState immediately after adding - should succeed
    boolean removed = parallel.removeState(1);
    assertThat(removed).isTrue();

    // Test removing non-existent sequence
    removed = parallel.removeState(99);
    assertThat(removed).isFalse();

    // Use iterator pattern: hasNext() + next()
    Map<Integer, StringBuilder> outputs = new HashMap<>();
    outputs.put(0, new StringBuilder());
    outputs.put(1, new StringBuilder());

    // Generate tokens - only seq 0 should generate since seq 1 was removed
    while (parallel.hasNext()) {
      LlamaOutput output = parallel.next();
      outputs.get(output.sequenceId()).append(output.text());
    }

    // Verify only seq 0 generated content
    assertThat(outputs.get(0).length()).isGreaterThan(0);
    assertThat(outputs.get(1).length()).isEqualTo(0);

    parallel.free();
    context.free();
    sampler.free();
    model.free();
  }

  @Test
  void test_batch_iterator_stop() throws Exception {
    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = new LlamaModel(arena, absolutePath, modelParameters);
    var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512).nSeqMax(2);
    var context = new LlamaContext(model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());

    var prompt = getPrompt(model, arena, buildMessages(arena, SYSTEM, "Count to 100"), contextParams);
    var state = ConversationState.create(arena, context, tokenizer, sampler, 0).setMaxTokens(100).initialize(prompt);

    var parallel = new BatchIterator(arena, context).addState(state);

    // Generate a few tokens then stop
    int tokenCount = 0;
    while (parallel.hasNext() && tokenCount < 5) {
      parallel.next();
      tokenCount++;
    }

    // Explicitly stop the iterator
    parallel.stop();

    // Verify hasNext returns false after stop
    assertThat(parallel.hasNext()).isFalse();
    assertThat(parallel.hasActiveConversations()).isFalse();

    parallel.free();
    context.free();
    sampler.free();
    model.free();
  }

  @Test
  void test_batch_iterator_duplicate_sequence_id_throws() throws Exception {
    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = new LlamaModel(arena, absolutePath, modelParameters);
    var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512).nSeqMax(2);
    var context = new LlamaContext(model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());

    var prompt = getPrompt(model, arena, buildMessages(arena, SYSTEM, "Hello"), contextParams);
    var state1 = ConversationState.create(arena, context, tokenizer, sampler, 0).initialize(prompt);
    var state2 = ConversationState.create(arena, context, tokenizer, sampler, 0).initialize(prompt); // Same seq ID!

    var parallel = new BatchIterator(arena, context).addState(state1);

    // Should throw when adding state with duplicate sequence ID
    assertThatThrownBy(() -> parallel.addState(state2))
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("Sequence ID 0 is already in use");

    parallel.free();
    context.free();
    sampler.free();
    model.free();
  }

  @Test
  void test_batch_iterator_different_context_throws() throws Exception {
    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = new LlamaModel(arena, absolutePath, modelParameters);
    var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512).nSeqMax(2);
    var context1 = new LlamaContext(model, contextParams);
    var context2 = new LlamaContext(model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer1 = new LlamaTokenizer(vocab, context1);
    var tokenizer2 = new LlamaTokenizer(vocab, context2);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());

    var prompt = getPrompt(model, arena, buildMessages(arena, SYSTEM, "Hello"), contextParams);
    var state1 = ConversationState.create(arena, context1, tokenizer1, sampler, 0).initialize(prompt);
    var state2 = ConversationState.create(arena, context2, tokenizer2, sampler, 1).initialize(prompt);

    var parallel = new BatchIterator(arena, context1).addState(state1);

    // Should throw when adding state with different context
    assertThatThrownBy(() -> parallel.addState(state2))
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("All conversation states must share the same LlamaContext");

    parallel.free();
    context1.free();
    context2.free();
    sampler.free();
    model.free();
  }

  @Test
  void test_batch_iterator_next_without_hasNext_throws() throws Exception {
    var modelParameters = new LlamaModelParams(arena);
    Path absolutePath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = new LlamaModel(arena, absolutePath, modelParameters);
    var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512).nSeqMax(2);
    var context = new LlamaContext(model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());

    var prompt = getPrompt(model, arena, buildMessages(arena, SYSTEM, "Hi"), contextParams);
    var state = ConversationState.create(arena, context, tokenizer, sampler, 0).setMaxTokens(1).initialize(prompt);

    var parallel = new BatchIterator(arena, context).addState(state);

    // Consume all outputs
    while (parallel.hasNext()) {
      parallel.next();
    }

    // Should throw when calling next() without hasNext()
    assertThatThrownBy(() -> parallel.next())
      .isInstanceOf(java.util.NoSuchElementException.class)
      .hasMessageContaining("No more outputs available");

    parallel.free();
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
