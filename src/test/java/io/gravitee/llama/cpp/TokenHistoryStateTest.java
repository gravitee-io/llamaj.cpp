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

import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_TO_DOWNLOAD;
import static io.gravitee.llama.cpp.LlamaCppTest.getModelPath;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Committed-token history invariant: {@code committedTokens().length == nPast} at all stable
 * points, through autoregressive AND speculative generation (including speculative rejection
 * rollbacks), single-stream and batched.
 *
 * @author GraviteeSource Team
 */
@Tag("integration")
class TokenHistoryStateTest extends LlamaCppTest {

  private static final String PROMPT = "The capital of France is";
  private static final int MAX_TOKENS = 24;

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

  private static int[] promptTokens(ConversationState state) {
    var r = state.getTokenized();
    int[] out = new int[r.size()];
    for (int i = 0; i < r.size(); i++) {
      out[i] = r.data().getAtIndex(JAVA_INT, i);
    }
    return out;
  }

  private static void assertHistoryInvariant(ConversationState state) {
    assertThat(state.committedTokens())
      .as("history.length == nPast")
      .hasSize(state.getNPast());
    assertThat(state.committedTokens()).startsWith(promptTokens(state));
  }

  @Test
  void history_tracks_npast_autoregressive() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var ctx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);

    var state = ConversationState.create(
      arena,
      ctx,
      new LlamaTokenizer(vocab, ctx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .initialize(PROMPT);

    var it = new DefaultLlamaIterator(state);
    while (it.hasNext()) {
      it.next();
      // Stable point between steps: every resident KV row is mirrored in the history.
      assertHistoryInvariant(state);
    }
    assertHistoryInvariant(state);
    assertThat(state.getNPast()).isGreaterThan(promptTokens(state).length);
  }

  @Test
  void history_tracks_npast_speculative_ngram_with_rollbacks() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var ctx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);

    // N-gram drafting: rounds routinely reject proposals, exercising the rollback truncation.
    var state = ConversationState.create(
      arena,
      ctx,
      new LlamaTokenizer(vocab, ctx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .setNgram(SpeculativeConfig.ngramGreedy(4, 2))
      .initialize(PROMPT);

    var it = new DefaultLlamaIterator(state);
    while (it.hasNext()) {
      it.next();
      assertHistoryInvariant(state);
    }
    assertHistoryInvariant(state);
  }

  @Test
  void history_tracks_npast_speculative_model_draft() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var ctx = track(new LlamaContext(arena, model, cp));
    var draftCtx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);

    var state = ConversationState.create(
      arena,
      ctx,
      new LlamaTokenizer(vocab, ctx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .setDraft(draftCtx, SpeculativeConfig.greedy(4))
      .initialize(PROMPT);

    var it = new DefaultLlamaIterator(state);
    while (it.hasNext()) {
      it.next();
      assertHistoryInvariant(state);
    }
    assertHistoryInvariant(state);
  }

  @Test
  void history_tracks_npast_in_batch_iterator() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena)
      .nCtx(512)
      .nBatch(512)
      .nUBatch(512)
      .nSeqMax(2);
    var ctx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);
    String[] prompts = { "The capital of France is", "Water boils at" };

    ConversationState[] states = new ConversationState[2];
    try (var it = new BatchIterator(arena, ctx)) {
      for (int i = 0; i < 2; i++) {
        states[i] = ConversationState.create(
          arena,
          ctx,
          new LlamaTokenizer(vocab, ctx),
          track(new LlamaSampler(arena).greedy()),
          i
        )
          .setMaxTokens(MAX_TOKENS)
          .initialize(prompts[i]);
        it.addState(states[i]);
      }
      while (it.hasNext()) {
        it.next();
      }
    }
    for (ConversationState s : states) {
      assertHistoryInvariant(s);
    }
  }
}
