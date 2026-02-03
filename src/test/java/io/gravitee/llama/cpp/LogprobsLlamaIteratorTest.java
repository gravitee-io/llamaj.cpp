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
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class LogprobsLlamaIteratorTest extends LlamaCppTest {

  private static Arena arena;

  @BeforeAll
  static void beforeAll() {
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
  void logprobs_are_null_when_disabled() {
    var model = new LlamaModel(
      arena,
      getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
      new LlamaModelParams(arena)
    );
    var contextParams = new LlamaContextParams(arena).noPerf(false);
    var context = new LlamaContext(arena, model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());
    var prompt = getPrompt(
      model,
      arena,
      buildMessages(arena, SYSTEM, "What is the capital of France?"),
      contextParams
    );

    var state = ConversationState.create(arena, context, tokenizer, sampler)
      .setMaxTokens(5)
      // topLogprobs not set — defaults to 0
      .initialize(prompt);

    var it = new DefaultLlamaIterator(state);

    List<LlamaOutput> outputs = it.stream().toList();

    assertThat(outputs).isNotEmpty();
    // Every token must have null logprobs when disabled
    outputs.forEach(output ->
      assertThat(output.logprobs())
        .as("logprobs should be null when topLogprobs == 0")
        .isNull()
    );

    context.free();
    sampler.free();
    model.free();
  }

  @Test
  void logprobs_are_present_for_every_token_when_enabled() {
    var model = new LlamaModel(
      arena,
      getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
      new LlamaModelParams(arena)
    );
    var contextParams = new LlamaContextParams(arena).noPerf(false);
    var context = new LlamaContext(arena, model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());
    var prompt = getPrompt(
      model,
      arena,
      buildMessages(arena, SYSTEM, "What is the capital of France?"),
      contextParams
    );

    int topN = 5;
    var state = ConversationState.create(arena, context, tokenizer, sampler)
      .setMaxTokens(10)
      .setTopLogprobs(topN)
      .initialize(prompt);

    var it = new DefaultLlamaIterator(state);

    List<LlamaOutput> outputs = it.stream().toList();

    assertThat(outputs).isNotEmpty();

    outputs.forEach(output -> {
      Logprobs logprobs = output.logprobs();

      assertThat(logprobs)
        .as("logprobs must be present for every token when topLogprobs > 0")
        .isNotNull();

      // Chosen token
      TokenLogprob chosen = logprobs.chosenToken();
      assertThat(chosen).isNotNull();
      assertThat(chosen.logprob())
        .as("chosen token logprob must be <= 0 (log of a probability)")
        .isLessThanOrEqualTo(0.0);
      assertThat(chosen.token()).isNotNull();

      // Top-N list
      List<TokenLogprob> topList = logprobs.topLogprobs();
      assertThat(topList).as("topLogprobs list must not be empty").isNotEmpty();
      assertThat(topList.size())
        .as(
          "topLogprobs list must contain at least topN=%d entries (chosen is always included)",
          topN
        )
        .isGreaterThanOrEqualTo(topN);

      // The chosen token must appear in the top list
      assertThat(topList)
        .as("chosen token must appear in topLogprobs list")
        .anyMatch(t -> t.tokenId() == chosen.tokenId());

      // All logprobs must be <= 0
      topList.forEach(t ->
        assertThat(t.logprob())
          .as("all logprobs must be <= 0")
          .isLessThanOrEqualTo(0.0)
      );

      // List must be sorted by descending logprob
      for (int i = 0; i < topList.size() - 1; i++) {
        assertThat(topList.get(i).logprob())
          .as("topLogprobs must be sorted by descending logprob")
          .isGreaterThanOrEqualTo(topList.get(i + 1).logprob());
      }
    });

    context.free();
    sampler.free();
    model.free();
  }

  @Test
  void chosen_token_logprob_matches_token_in_output() {
    var model = new LlamaModel(
      arena,
      getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
      new LlamaModelParams(arena)
    );
    var contextParams = new LlamaContextParams(arena).noPerf(false);
    var context = new LlamaContext(arena, model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(42);
    var prompt = getPrompt(
      model,
      arena,
      buildMessages(arena, SYSTEM, "What is the capital of France?"),
      contextParams
    );

    var state = ConversationState.create(arena, context, tokenizer, sampler)
      .setMaxTokens(5)
      .setTopLogprobs(3)
      .initialize(prompt);

    var it = new DefaultLlamaIterator(state);

    it
      .stream()
      .forEach(output -> {
        assertThat(output.logprobs()).isNotNull();
        TokenLogprob chosen = output.logprobs().chosenToken();

        // The chosen token's text must match the output content
        assertThat(output.content())
          .as("output content must match the chosen token text")
          .isEqualTo(chosen.token());
      });

    context.free();
    sampler.free();
    model.free();
  }

  @AfterAll
  static void afterAll() {
    arena = null;
    LlamaRuntime.llama_backend_free();
  }
}
