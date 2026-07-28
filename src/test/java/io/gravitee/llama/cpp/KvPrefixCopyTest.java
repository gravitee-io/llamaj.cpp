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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

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
 * Cross-sequence KV prefix publishing: {@link LlamaMemory#copyPrefix(int, int, int, int)} shares a
 * resident prefix with a second sequence so it prefills only the suffix.
 *
 * <p>Where {@code PrefixReuseTest} covers reuse across turns of the <em>same</em> sequence, this
 * covers reuse across <em>different</em> ones — including copying from a sequence that is still
 * generating, which is what lets concurrent conversations behind a shared system prompt prefill it
 * once between them. Policy (which sequence, eviction, cache keys) lives in the serving layer;
 * this is the mechanism it stands on.
 *
 * @author GraviteeSource Team
 */
@Tag("integration")
class KvPrefixCopyTest extends LlamaCppTest {

  private static final String PROMPT = "The capital of France is";
  private static final int MAX_TOKENS = 8;

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

  /**
   * The core claim: with sequence 0 mid-generation, an identical prompt on sequence 1 decodes
   * exactly one prompt token — the final one, always re-decoded for its logits — instead of the
   * whole prompt, and produces byte-identical greedy output. The donor is left untouched.
   */
  @Test
  void prefix_copied_from_a_generating_sequence_prefills_only_the_suffix() {
    var ctx = newContext();
    var tokenizer = new LlamaTokenizer(new LlamaVocab(ctx.getModel()), ctx);
    int[] tokens = tokensOf(tokenizer, PROMPT);

    // Sequence 0 — cold, and deliberately left mid-generation so the copy reads a live sequence.
    var donor = newState(ctx, tokenizer, 0).initialize(PROMPT);
    var donorIterator = new DefaultLlamaIterator(donor);
    List<String> donorPieces = new ArrayList<>();
    for (int i = 0; i < 3 && donorIterator.hasNext(); i++) {
      donorPieces.add(donorIterator.next().content());
    }
    assertThat(donor.isFinished()).isFalse();
    int donorSpan = ctx.getMemory().posMax(0);
    assertThat(donorSpan).isGreaterThanOrEqualTo(tokens.length - 1);

    // Publish its prompt prefix onto sequence 1. committedTokens() is exactly what is resident.
    int[] committed = donor.committedTokens();
    int shared = commonPrefix(committed, tokens);
    int reuse = ctx.getMemory().copyPrefix(0, 1, shared, tokens.length);

    assertThat(reuse).isEqualTo(tokens.length - 1);
    assertThat(ctx.getMemory().posMax(1)).isEqualTo(reuse - 1);
    // Zero-copy: publishing to sequence 1 did not disturb sequence 0.
    assertThat(ctx.getMemory().posMax(0)).isEqualTo(donorSpan);

    // Sequence 1 — warm: only the final prompt token is decoded.
    var warm = newState(ctx, tokenizer, 1).initialize(PROMPT, reuse);
    var warmIterator = new DefaultLlamaIterator(warm);
    long before = decodedTokens(ctx);
    boolean hasNext = warmIterator.hasNext();
    long warmPrefill = decodedTokens(ctx) - before;
    String warmText = drain(warmIterator, hasNext);

    assertThat(warm.isPrefixReuseHonored()).isTrue();
    assertThat(warmPrefill).isEqualTo(1);

    // The donor keeps generating correctly after having been copied from.
    while (donorIterator.hasNext()) {
      donorPieces.add(donorIterator.next().content());
    }
    String donorText = String.join("", donorPieces);

    // A cold run of the same prompt on a fresh context is the reference for both.
    String coldText = runCold(PROMPT);
    assertThat(warmText).isEqualTo(coldText);
    assertThat(donorText).isEqualTo(coldText);
  }

  /** A copy too short to be worth anything still leaves the destination clean and empty. */
  @Test
  void trivial_match_copies_nothing_and_clears_the_destination() {
    var ctx = newContext();
    var tokenizer = new LlamaTokenizer(new LlamaVocab(ctx.getModel()), ctx);

    var donor = newState(ctx, tokenizer, 0).initialize(PROMPT);
    assertThat(new DefaultLlamaIterator(donor).hasNext()).isTrue();

    // Seed sequence 1 with unrelated rows, then publish a zero-length prefix onto it.
    var stale = newState(ctx, tokenizer, 1).initialize("Once upon a time in");
    assertThat(new DefaultLlamaIterator(stale).hasNext()).isTrue();
    assertThat(ctx.getMemory().posMax(1)).isGreaterThanOrEqualTo(0);

    assertThat(ctx.getMemory().copyPrefix(0, 1, 0, 12)).isZero();
    assertThat(ctx.getMemory().posMax(1)).isEqualTo(-1);

    // A single-token prompt can never reuse anything: its only token needs re-decoding.
    assertThat(ctx.getMemory().copyPrefix(0, 1, 5, 1)).isZero();

    assertThatThrownBy(() -> ctx.getMemory().copyPrefix(0, 0, 4, 10))
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("must differ");
  }

  // ---------------------------------------------------------------------------------------

  private LlamaContext newContext() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var params = new LlamaContextParams(arena)
      .nCtx(2048)
      .nBatch(512)
      .nUBatch(512)
      .nSeqMax(4)
      // REQUIRED for copyPrefix: one shared cell pool, so cells carry a set of sequence ids and
      // seq_cp is metadata-only. Without it each sequence owns a stream, seq_cp must copy buffer
      // data, and a partial range aborts the process.
      .kvUnified(true)
      .noPerf(false);
    return track(new LlamaContext(arena, model, params));
  }

  /**
   * The guard that keeps a misconfiguration from killing the JVM. On a non-unified context
   * llama.cpp would trip GGML_ASSERT(is_full) inside seq_cp and abort() the process; copyPrefix
   * must refuse first.
   */
  @Test
  void a_non_unified_context_is_refused_rather_than_aborting() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var params = new LlamaContextParams(arena)
      .nCtx(512)
      .nBatch(256)
      .nUBatch(256)
      .nSeqMax(2)
      .kvUnified(false);
    var ctx = track(new LlamaContext(arena, model, params));

    assertThat(ctx.isKvUnified()).isFalse();
    assertThatThrownBy(() -> ctx.getMemory().copyPrefix(0, 1, 4, 10))
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("unified KV cache");
  }

  private ConversationState newState(
    LlamaContext ctx,
    LlamaTokenizer tokenizer,
    int sequenceId
  ) {
    return ConversationState.create(
      arena,
      ctx,
      tokenizer,
      track(new LlamaSampler(arena).greedy()),
      sequenceId
    )
      .setMaxTokens(MAX_TOKENS)
      // Without this the iterator wipes the sequence on finish and the published prefix becomes
      // a dangling claim.
      .setRetainKv(true);
  }

  private String runCold(String prompt) {
    var ctx = newContext();
    var tokenizer = new LlamaTokenizer(new LlamaVocab(ctx.getModel()), ctx);
    var state = newState(ctx, tokenizer, 0).initialize(prompt);
    var iterator = new DefaultLlamaIterator(state);
    return drain(iterator, iterator.hasNext());
  }

  private static String drain(DefaultLlamaIterator iterator, boolean hasNext) {
    List<String> pieces = new ArrayList<>();
    while (hasNext) {
      pieces.add(iterator.next().content());
      hasNext = iterator.hasNext();
    }
    return String.join("", pieces);
  }

  private static int commonPrefix(int[] a, int[] b) {
    int n = Math.min(a.length, b.length);
    int i = 0;
    while (i < n && a[i] == b[i]) {
      i++;
    }
    return i;
  }

  private static int[] tokensOf(LlamaTokenizer tokenizer, String prompt) {
    var response = tokenizer.tokenize(arena, prompt);
    int[] out = new int[response.size()];
    for (int i = 0; i < response.size(); i++) {
      out[i] = response.data().getAtIndex(JAVA_INT, i);
    }
    return out;
  }

  private static long decodedTokens(LlamaContext ctx) {
    var p = ctx.getPerformance(arena);
    return (long) p.promptTokensEvaluated() + p.tokensGenerated();
  }
}
