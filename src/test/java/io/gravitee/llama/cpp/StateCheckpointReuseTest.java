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

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Prefix reuse on a hybrid model through {@link StateCheckpoints}: a warm turn that diverges
 * before the end of the resident state resumes from a checkpoint instead of re-reading the
 * whole prompt, with the same greedy output as a cold prefill.
 *
 * @author GraviteeSource Team
 */
@Tag("integration")
class StateCheckpointReuseTest extends LlamaCppTest {

  private static final int MAX_TOKENS = 16;

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

  @Test
  void warm_turn_resumes_from_the_prompt_end_checkpoint() {
    var model = load(HYBRID_MODEL_PATH, HYBRID_MODEL_TO_DOWNLOAD);
    assertThat(model.hasRecurrentState()).isTrue();
    var ctx = context(model);
    var tokenizer = tokenizer(ctx);
    assertThat(ctx.stateCheckpoints().isEnabled()).isTrue();

    String turn1 = facts(0, 20);
    String turn2 = turn1 + notes(0, 30);
    int[] tokens1 = tokensOf(tokenizer, turn1);
    int[] tokens2 = tokensOf(tokenizer, turn2);
    assertThat(lcp(tokens1, tokens2)).isEqualTo(tokens1.length);

    var first = run(newState(ctx, tokenizer).initialize(turn1));
    assertThat(ctx.stateCheckpoints().positions(0)).containsExactly(
      tokens1.length
    );
    assertThat(ctx.stateCheckpoints().byteSize()).isPositive();

    int requested = lcp(first.committedTokens(), tokens2);
    assertThat(requested).isLessThan(first.getNPast());
    assertThat(tokens2.length - tokens1.length).isGreaterThan(200);

    var warm = newState(ctx, tokenizer).initialize(turn2, requested);
    long decoded = prefill(ctx, warm);

    assertThat(warm.getReusedPrefixTokens()).isEqualTo(tokens1.length);
    assertThat(warm.isPrefixReuseHonored()).isEqualTo(
      requested == tokens1.length
    );
    assertThat(decoded).isEqualTo(tokens2.length - tokens1.length);
    assertThat(warm.committedTokens()).isEqualTo(cold(model, turn2));
    assertThat(ctx.stateCheckpoints().positions(0)).containsExactly(
      tokens1.length,
      tokens2.length
    );
  }

  @Test
  void interval_checkpoints_cover_a_divergence_inside_the_previous_prompt() {
    var model = load(HYBRID_MODEL_PATH, HYBRID_MODEL_TO_DOWNLOAD);
    var ctx = context(model);
    ctx.stateCheckpoints().setMaxPerSequence(16).setInterval(64);
    var tokenizer = tokenizer(ctx);

    String turn1 = facts(0, 30);
    int[] tokens1 = tokensOf(tokenizer, turn1);
    var first = run(newState(ctx, tokenizer).initialize(turn1));
    var expected = IntStream.iterate(64, p -> p < tokens1.length, p -> p + 64)
      .boxed()
      .collect(Collectors.toList());
    expected.add(tokens1.length);
    assertThat(ctx.stateCheckpoints().positions(0)).isEqualTo(expected);

    String turn2 = facts(0, 12) + notes(0, 10);
    int[] tokens2 = tokensOf(tokenizer, turn2);
    int requested = lcp(first.committedTokens(), tokens2);
    int checkpoint = (requested / 64) * 64;
    assertThat(checkpoint).isPositive();

    var warm = newState(ctx, tokenizer).initialize(turn2, requested);
    long decoded = prefill(ctx, warm);

    assertThat(warm.getReusedPrefixTokens()).isEqualTo(checkpoint);
    assertThat(warm.isPrefixReuseHonored()).isEqualTo(requested == checkpoint);
    assertThat(decoded).isEqualTo(tokens2.length - checkpoint);
    assertThat(warm.committedTokens()).isEqualTo(cold(model, turn2));
  }

  @Test
  void on_device_checkpoint_resumes_a_warm_turn() {
    var model = load(HYBRID_MODEL_PATH, HYBRID_MODEL_TO_DOWNLOAD);
    var ctx = context(model);
    ctx.stateCheckpoints().setOnDevice(true);
    var tokenizer = tokenizer(ctx);

    String turn1 = facts(0, 20);
    String turn2 = turn1 + notes(0, 30);
    int[] tokens1 = tokensOf(tokenizer, turn1);
    int[] tokens2 = tokensOf(tokenizer, turn2);

    var first = run(newState(ctx, tokenizer).initialize(turn1));
    assertThat(ctx.stateCheckpoints().positions(0)).containsExactly(
      tokens1.length
    );
    assertThat(ctx.stateCheckpoints().byteSize()).isLessThan(64 * 1024);

    int requested = lcp(first.committedTokens(), tokens2);
    assertThat(requested).isLessThan(first.getNPast());
    var warm = newState(ctx, tokenizer).initialize(turn2, requested);
    long decoded = prefill(ctx, warm);

    assertThat(warm.getReusedPrefixTokens()).isEqualTo(tokens1.length);
    assertThat(decoded).isEqualTo(tokens2.length - tokens1.length);
    assertThat(warm.committedTokens()).isEqualTo(cold(model, turn2));
    assertThat(ctx.stateCheckpoints().positions(0)).containsExactly(
      tokens2.length
    );

    String turn3 = turn2 + facts(20, 30);
    int[] tokens3 = tokensOf(tokenizer, turn3);
    int requested3 = lcp(warm.committedTokens(), tokens3);
    assertThat(requested3).isLessThan(warm.getNPast());
    var third = newState(ctx, tokenizer).initialize(turn3, requested3);
    long decoded3 = prefill(ctx, third);

    assertThat(third.getReusedPrefixTokens()).isEqualTo(tokens2.length);
    assertThat(decoded3).isEqualTo(tokens3.length - tokens2.length);
    assertThat(third.committedTokens()).isEqualTo(cold(model, turn3));
  }

  @Test
  void on_device_falls_back_cold_when_the_latest_checkpoint_does_not_match() {
    var model = load(HYBRID_MODEL_PATH, HYBRID_MODEL_TO_DOWNLOAD);
    var ctx = context(model);
    ctx.stateCheckpoints().setOnDevice(true);
    var tokenizer = tokenizer(ctx);

    var first = run(newState(ctx, tokenizer).initialize(facts(0, 20)));

    String turn2 = facts(0, 10) + notes(0, 10);
    int[] tokens2 = tokensOf(tokenizer, turn2);
    int requested = lcp(first.committedTokens(), tokens2);
    var warm = newState(ctx, tokenizer).initialize(turn2, requested);
    long decoded = prefill(ctx, warm);

    assertThat(warm.isPrefixReuseHonored()).isFalse();
    assertThat(warm.getReusedPrefixTokens()).isZero();
    assertThat(decoded).isEqualTo(tokens2.length);
    assertThat(warm.committedTokens()).isEqualTo(cold(model, turn2));
  }

  @Test
  void no_matching_checkpoint_falls_back_to_a_cold_prefill() {
    var model = load(HYBRID_MODEL_PATH, HYBRID_MODEL_TO_DOWNLOAD);
    var ctx = context(model);
    var tokenizer = tokenizer(ctx);

    String turn1 = facts(0, 20);
    var first = run(newState(ctx, tokenizer).initialize(turn1));

    String turn2 = facts(0, 10) + notes(0, 10);
    int[] tokens2 = tokensOf(tokenizer, turn2);
    int requested = lcp(first.committedTokens(), tokens2);
    assertThat(requested).isPositive();

    var warm = newState(ctx, tokenizer).initialize(turn2, requested);
    long decoded = prefill(ctx, warm);

    assertThat(warm.isPrefixReuseHonored()).isFalse();
    assertThat(warm.getReusedPrefixTokens()).isZero();
    assertThat(decoded).isEqualTo(tokens2.length);
    assertThat(warm.committedTokens()).isEqualTo(cold(model, turn2));
  }

  @Test
  void keeps_at_most_n_checkpoints_and_zero_disables_them() {
    var model = load(HYBRID_MODEL_PATH, HYBRID_MODEL_TO_DOWNLOAD);
    var ctx = context(model);
    ctx.stateCheckpoints().setMaxPerSequence(2);
    var tokenizer = tokenizer(ctx);

    int[] lengths = IntStream.of(4, 6, 8)
      .map(n -> {
        String prompt = facts(0, n);
        run(newState(ctx, tokenizer).initialize(prompt));
        return tokensOf(tokenizer, prompt).length;
      })
      .toArray();
    assertThat(ctx.stateCheckpoints().positions(0)).containsExactly(
      lengths[1],
      lengths[2]
    );

    ctx.stateCheckpoints().setMaxPerSequence(0);
    assertThat(ctx.stateCheckpoints().size(0)).isZero();
    assertThat(ctx.stateCheckpoints().byteSize()).isZero();

    String turn1 = facts(0, 20);
    var first = run(newState(ctx, tokenizer).initialize(turn1));
    assertThat(ctx.stateCheckpoints().size(0)).isZero();

    String turn2 = turn1 + notes(0, 10);
    int[] tokens2 = tokensOf(tokenizer, turn2);
    int requested = lcp(first.committedTokens(), tokens2);
    var warm = newState(ctx, tokenizer).initialize(turn2, requested);
    long decoded = prefill(ctx, warm);

    assertThat(warm.isPrefixReuseHonored()).isFalse();
    assertThat(warm.getReusedPrefixTokens()).isZero();
    assertThat(decoded).isEqualTo(tokens2.length);
  }

  @Test
  void attention_models_never_take_checkpoints() {
    var model = load(MODEL_PATH, MODEL_TO_DOWNLOAD);
    assertThat(model.hasRecurrentState()).isFalse();
    var ctx = context(model);
    var tokenizer = tokenizer(ctx);
    assertThat(ctx.stateCheckpoints().isSupported()).isFalse();
    assertThat(ctx.stateCheckpoints().isEnabled()).isFalse();

    String turn1 = facts(0, 20);
    var first = run(newState(ctx, tokenizer).initialize(turn1));
    String turn2 = turn1 + notes(0, 10);
    int[] tokens2 = tokensOf(tokenizer, turn2);
    int requested = lcp(first.committedTokens(), tokens2);
    var warm = newState(ctx, tokenizer).initialize(turn2, requested);
    long decoded = prefill(ctx, warm);

    assertThat(warm.isPrefixReuseHonored()).isTrue();
    assertThat(warm.getReusedPrefixTokens()).isEqualTo(requested);
    assertThat(decoded).isEqualTo(tokens2.length - requested);
    assertThat(ctx.stateCheckpoints().size(0)).isZero();
    assertThat(ctx.stateCheckpoints().byteSize()).isZero();
  }

  private LlamaModel load(String path, String url) {
    return track(
      new LlamaModel(
        arena,
        getModelPath(path, url),
        new LlamaModelParams(arena)
      )
    );
  }

  private LlamaContext context(LlamaModel model) {
    var params = new LlamaContextParams(arena)
      .nCtx(2048)
      .nBatch(512)
      .nUBatch(512)
      .noPerf(false);
    return track(new LlamaContext(arena, model, params));
  }

  private static LlamaTokenizer tokenizer(LlamaContext ctx) {
    return new LlamaTokenizer(new LlamaVocab(ctx.getModel()), ctx);
  }

  private ConversationState newState(
    LlamaContext ctx,
    LlamaTokenizer tokenizer
  ) {
    return ConversationState.create(
      arena,
      ctx,
      tokenizer,
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .setRetainKv(true);
  }

  private static ConversationState run(ConversationState state) {
    try (var it = new DefaultLlamaIterator(state)) {
      while (it.hasNext()) {
        it.next();
      }
    }
    return state;
  }

  private static long prefill(LlamaContext ctx, ConversationState state) {
    try (var it = new DefaultLlamaIterator(state)) {
      long before = decodedTokens(ctx);
      boolean hasNext = it.hasNext();
      long decoded = decodedTokens(ctx) - before;
      if (hasNext) {
        it.next();
        while (it.hasNext()) {
          it.next();
        }
      }
      return decoded;
    }
  }

  private int[] cold(LlamaModel model, String prompt) {
    var ctx = context(model);
    return run(
      newState(ctx, tokenizer(ctx)).initialize(prompt)
    ).committedTokens();
  }

  private static long decodedTokens(LlamaContext ctx) {
    var p = ctx.getPerformance(arena);
    return (long) p.promptTokensEvaluated() + p.tokensGenerated();
  }

  private static String facts(int from, int to) {
    return IntStream.range(from, to)
      .mapToObj(i ->
        "Fact %d: the warehouse in district %d stores %d crates of item %d.\n".formatted(
          i,
          (i * 7) % 13,
          i * 37 + 11,
          i * 3 + 2
        )
      )
      .collect(Collectors.joining());
  }

  private static String notes(int from, int to) {
    return IntStream.range(from, to)
      .mapToObj(i ->
        "Note %d: shipment %d left dock %d on day %d with %d pallets.\n".formatted(
          i,
          i * 19 + 5,
          i % 4,
          i * 2 + 1,
          i * 11 + 3
        )
      )
      .collect(Collectors.joining());
  }

  private static int[] tokensOf(LlamaTokenizer tokenizer, String prompt) {
    var r = tokenizer.tokenize(arena, prompt);
    return r
      .data()
      .asSlice(0, r.size() * JAVA_INT.byteSize())
      .toArray(JAVA_INT);
  }

  private static int lcp(int[] a, int[] b) {
    int mismatch = Arrays.mismatch(a, b);
    return mismatch < 0 ? a.length : mismatch;
  }
}
