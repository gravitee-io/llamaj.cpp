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
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * KV prefix reuse: {@code initialize(prompt, reusePrefixTokens)} + {@code setRetainKv} /
 * {@code removeState(id, keepKv)} let a follow-up prompt sharing a token prefix with a previous
 * one on the same sequence skip re-decoding the shared prefix.
 *
 * @author GraviteeSource Team
 */
@Tag("integration")
class PrefixReuseTest extends LlamaCppTest {

  private static final String PROMPT = "The capital of France is";
  private static final String EXTENSION = " Paris. The capital of Germany is";
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

  private static int[] tokensOf(LlamaTokenizer tokenizer, String prompt) {
    var r = tokenizer.tokenize(arena, prompt);
    int[] out = new int[r.size()];
    for (int i = 0; i < r.size(); i++) {
      out[i] = r.data().getAtIndex(JAVA_INT, i);
    }
    return out;
  }

  private static int lcp(int[] a, int[] b) {
    int n = Math.min(a.length, b.length);
    int i = 0;
    while (i < n && a[i] == b[i]) {
      i++;
    }
    return i;
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
    ).setMaxTokens(MAX_TOKENS);
  }

  private static List<LlamaOutput> collect(DefaultLlamaIterator it) {
    List<LlamaOutput> out = new ArrayList<>();
    it.stream().forEach(out::add);
    return out;
  }

  private static String text(List<LlamaOutput> outputs) {
    return outputs
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);
  }

  @Test
  void prefix_reuse_matches_cold_prefill_and_clamps_identical_prompt() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var reuseCtx = track(new LlamaContext(arena, model, cp));
    var refCtx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);
    var reuseTok = new LlamaTokenizer(vocab, reuseCtx);
    var refTok = new LlamaTokenizer(vocab, refCtx);

    int[] tokensA = tokensOf(reuseTok, PROMPT);
    String combined = PROMPT + EXTENSION;
    int[] tokensAB = tokensOf(reuseTok, combined);
    int prefix = lcp(tokensA, tokensAB);
    assertThat(prefix).isGreaterThan(0);

    // (a) Cold greedy run of prompt A on seq 0, retaining the KV on finish.
    var stateA = newState(reuseCtx, reuseTok)
      .setRetainKv(true)
      .initialize(PROMPT);
    List<LlamaOutput> outputsA = collect(new DefaultLlamaIterator(stateA));
    String outA = text(outputsA);
    assertThat(outA).isNotBlank();
    // The committed history is exactly the KV-resident tokens: prompt + decoded generations.
    assertThat(stateA.committedTokens()).hasSize(stateA.getNPast());
    assertThat(stateA.committedTokens()).startsWith(tokensA);
    String firstPieceA = outputsA.get(0).content();

    // (c) Identical prompt with reuse == tokenized.size(): the clamp re-decodes only the last
    // prompt token; the first sampled token must match the cold run's.
    var stateC = newState(reuseCtx, reuseTok)
      .setRetainKv(true)
      .initialize(PROMPT, tokensA.length);
    assertThat(stateC.getReusePrefixTokens()).isEqualTo(tokensA.length - 1);
    try (var itC = new DefaultLlamaIterator(stateC)) {
      assertThat(itC.hasNext()).isTrue();
      assertThat(itC.next().content()).isEqualTo(firstPieceA);
    }

    // Reference: cold full prefill of the combined prompt on a fresh context.
    var refState = newState(refCtx, refTok).initialize(combined);
    String refOut = text(collect(new DefaultLlamaIterator(refState)));

    // (b) Prefix reuse: same seq id, combined prompt, reuse the common token prefix.
    var stateB = newState(reuseCtx, reuseTok)
      .setRetainKv(true)
      .initialize(combined, prefix);
    assertThat(stateB.getReusePrefixTokens()).isEqualTo(
      Math.min(prefix, tokensAB.length - 1)
    );
    String reuseOut = text(collect(new DefaultLlamaIterator(stateB)));

    System.out.println("cold : " + refOut);
    System.out.println("reuse: " + reuseOut);
    assertThat(reuseOut).isEqualTo(refOut);
    assertThat(stateB.committedTokens()).startsWith(tokensAB);
    assertThat(stateB.committedTokens()).hasSize(stateB.getNPast());
  }

  @Test
  void removeState_keepKv_controls_kv_residency() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var ctx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, ctx);

    // keepKv = true: the sequence's rows stay resident after removal.
    try (var it = new BatchIterator(arena, ctx)) {
      it.addState(newState(ctx, tokenizer).initialize(PROMPT));
      for (int i = 0; i < 4 && it.hasNext(); i++) {
        it.next();
      }
      assertThat(it.removeState(0, true)).isTrue();
      assertThat(ctx.getMemory().posMax(0)).isGreaterThanOrEqualTo(0);
    }
    // A keepKv-removed sequence is unregistered, so iterator teardown (stop/free) does not
    // touch it: its rows survive for a later prefix-reuse initialization.
    assertThat(ctx.getMemory().posMax(0)).isGreaterThanOrEqualTo(0);
    ctx.getMemory().seqRm(0, -1, -1);
    assertThat(ctx.getMemory().posMax(0)).isEqualTo(-1);

    // keepKv = false: forces the wipe even for a retainKv-marked state.
    try (var it = new BatchIterator(arena, ctx)) {
      it.addState(
        newState(ctx, tokenizer).setRetainKv(true).initialize(PROMPT)
      );
      for (int i = 0; i < 4 && it.hasNext(); i++) {
        it.next();
      }
      assertThat(it.removeState(0, false)).isTrue();
      assertThat(ctx.getMemory().posMax(0)).isEqualTo(-1);
    }
  }

  /**
   * MTP-seed contract under partial prefill: the MTP flavour seeds its head from
   * {@code context.getEmbeddingsIth(-1)} after {@code processPrompt} (the last output row —
   * a positive index maps through {@code output_ids[token_index]} and is NOT the output row
   * after a prompt prefill, where only the final token has an output). With prefix reuse only
   * the suffix rows are decoded, but the last output row must still be the last prompt token's
   * hidden — identical (within numeric tolerance) to a cold full prefill. An MTP-capable
   * (nextn-headed) tiny model is not available in CI, so this verifies the seed-row property
   * itself rather than an end-to-end MTP round.
   */
  @Test
  void partial_prefill_keeps_last_prompt_token_embedding_row_0() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var coldCtx = track(new LlamaContext(arena, model, cp));
    var reuseCtx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);
    var coldTok = new LlamaTokenizer(vocab, coldCtx);
    var reuseTok = new LlamaTokenizer(vocab, reuseCtx);
    LlamaRuntime.llama_set_embeddings(coldCtx.segment, true);
    LlamaRuntime.llama_set_embeddings(reuseCtx.segment, true);

    String combined = PROMPT + EXTENSION;
    int[] tokensA = tokensOf(reuseTok, PROMPT);
    int[] tokensAB = tokensOf(reuseTok, combined);
    int prefix = lcp(tokensA, tokensAB);

    // Cold full prefill of the combined prompt: read row 0 right after prompt processing.
    var coldState = newState(coldCtx, coldTok).initialize(combined);
    float[] coldSeed;
    String coldFirst;
    try (var it = new DefaultLlamaIterator(coldState)) {
      assertThat(it.hasNext()).isTrue();
      coldSeed = coldCtx.getEmbeddingsIth(-1);
      coldFirst = it.next().content();
    }

    // Warm the reuse context with prompt A, then partial-prefill the combined prompt.
    var warm = newState(reuseCtx, reuseTok)
      .setRetainKv(true)
      .initialize(PROMPT);
    collect(new DefaultLlamaIterator(warm));
    var reuseState = newState(reuseCtx, reuseTok)
      .setRetainKv(true)
      .initialize(combined, prefix);
    float[] reuseSeed;
    String reuseFirst;
    try (var it = new DefaultLlamaIterator(reuseState)) {
      assertThat(it.hasNext()).isTrue();
      reuseSeed = reuseCtx.getEmbeddingsIth(-1);
      reuseFirst = it.next().content();
    }

    assertThat(reuseFirst).isEqualTo(coldFirst);
    assertThat(reuseSeed).hasSameSizeAs(coldSeed);
    // Cosine similarity ~1: same row content up to batch-shape numeric noise.
    assertThat(cosine(coldSeed, reuseSeed)).isGreaterThan(0.999);
  }

  /**
   * MTP warm turns must be O(suffix): the target keeps embeddings enabled for seed extraction,
   * and llama.cpp forces every batch token to an output row when embeddings are on — before the
   * fix this made every prefill pay O(prompt) lm_head + embedding extraction, and a warm turn's
   * decode work was proportional to the full context. Assert (via the context's native decode
   * counters, not wall-clock) that a warm MTP prefill decodes exactly the un-reused suffix.
   * Uses the same fake-MTP-context trick as {@code MtpEagle3SpeculativeTest}: only the prefill
   * step runs (a single {@code hasNext()}), which never touches the nextn head graph.
   */
  @Test
  void mtp_warm_turn_prefill_work_is_proportional_to_suffix() {
    org.junit.jupiter.api.Assumptions.assumeTrue(LlamaExt.available());
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena)
      .nCtx(512)
      .nBatch(512)
      .nUBatch(512)
      .noPerf(false);
    var ctx = track(new LlamaContext(arena, model, cp));
    var mtpish = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, ctx);

    String combined = PROMPT + EXTENSION;
    int[] tokensA = tokensOf(tokenizer, PROMPT);
    int[] tokensAB = tokensOf(tokenizer, combined);
    int prefix = lcp(tokensA, tokensAB);
    assertThat(prefix).isGreaterThan(0);

    // Turn 1 (cold): MTP state, prefill only, KV retained.
    var s1 = newState(ctx, tokenizer)
      .setRetainKv(true)
      .setMtp(mtpish, SpeculativeConfig.greedy(4))
      .initialize(PROMPT);
    long before1 = decodedTokens(ctx);
    var it1 = new DefaultLlamaIterator(s1);
    assertThat(it1.hasNext()).isTrue();
    long cold = decodedTokens(ctx) - before1;
    // llama_perf floors empty counters at 1 (max(1, n)), so the first delta on a fresh
    // context under-reads by up to 1; subsequent deltas (the warm one below) are exact.
    assertThat(cold).isBetween(
      (long) tokensA.length - 1,
      (long) tokensA.length
    );

    // Turn 2 (warm): same sequence, combined prompt, reuse the common prefix.
    var s2 = newState(ctx, tokenizer)
      .setRetainKv(true)
      .setMtp(mtpish, SpeculativeConfig.greedy(4))
      .initialize(combined, prefix);
    long before2 = decodedTokens(ctx);
    var it2 = new DefaultLlamaIterator(s2);
    assertThat(it2.hasNext()).isTrue();
    long warm = decodedTokens(ctx) - before2;
    // O(suffix): only the un-reused prompt tokens are decoded, prefix reuse was honored.
    assertThat(s2.isPrefixReuseHonored()).isTrue();
    assertThat(warm).isEqualTo(tokensAB.length - prefix);
    String warmFirst = s2.getPiece();

    // Correctness of the split final-token MTP prefill: a plain cold greedy run of the
    // combined prompt on a fresh context samples the same first token.
    var refCtx = track(new LlamaContext(arena, model, cp));
    var refState = newState(
      refCtx,
      new LlamaTokenizer(vocab, refCtx)
    ).initialize(combined);
    try (var refIt = new DefaultLlamaIterator(refState)) {
      assertThat(refIt.hasNext()).isTrue();
      assertThat(warmFirst).isEqualTo(refState.getPiece());
    }
    // And the MTP seed row is still readable (embeddings restored, last row = last prompt token).
    assertThat(ctx.getEmbeddingsIth(-1)).hasSize(model.nEmbdOut());

    s1.freeSpeculativeScratch();
    s2.freeSpeculativeScratch();
  }

  /** Total tokens decoded on {@code ctx} so far (native perf counters: prompt + single-token). */
  private static long decodedTokens(LlamaContext ctx) {
    var p = ctx.getPerformance(arena);
    return (long) p.promptTokensEvaluated() + p.tokensGenerated();
  }

  private static double cosine(float[] a, float[] b) {
    double dot = 0,
      na = 0,
      nb = 0;
    for (int i = 0; i < a.length; i++) {
      dot += (double) a[i] * b[i];
      na += (double) a[i] * a[i];
      nb += (double) b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }
}
