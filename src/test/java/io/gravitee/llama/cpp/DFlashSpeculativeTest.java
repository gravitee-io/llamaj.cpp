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
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.gravitee.llama.cpp.draft.DFlashDraft;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Tests for {@link ConversationState#setDFlash}. The decoding cases run only when their target and
 * draft variables point at a target GGUF and a block-diffusion draft converted against it:
 * {@code LLAMAJ_DFLASH_TARGET} with {@code LLAMAJ_DFLASH_DRAFT}, {@code LLAMAJ_DFLASH2_DRAFT} or
 * {@code LLAMAJ_LILICORR_DRAFT}, and {@code LLAMAJ_DSPARK_TARGET} with {@code LLAMAJ_DSPARK_DRAFT}.
 */
class DFlashSpeculativeTest extends LlamaCppTest {

  private static final String PROMPT =
    "Count from one to fifty in words, separated by commas: one, two, three,";
  private static final String OTHER_PROMPT =
    "List the days of the week, separated by commas: Monday, Tuesday,";
  private static final int MAX_TOKENS = 96;

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

  static Stream<Arguments> drafts() {
    return Stream.of(
      Arguments.of(
        "LLAMAJ_DFLASH_TARGET",
        "LLAMAJ_DFLASH_DRAFT",
        DFlashDraft.Kind.DFLASH
      ),
      Arguments.of(
        "LLAMAJ_DSPARK_TARGET",
        "LLAMAJ_DSPARK_DRAFT",
        DFlashDraft.Kind.DSPARK
      ),
      Arguments.of(
        "LLAMAJ_DFLASH_TARGET",
        "LLAMAJ_DFLASH2_DRAFT",
        DFlashDraft.Kind.DFLASH2
      ),
      Arguments.of(
        "LLAMAJ_DFLASH_TARGET",
        "LLAMAJ_LILICORR_DRAFT",
        DFlashDraft.Kind.LILICORR
      )
    );
  }

  @Test
  void setDFlash_rejects_a_model_that_is_not_a_dflash_draft() {
    var model = track(
      new LlamaModel(
        arena,
        getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
        new LlamaModelParams(arena)
      )
    );
    var params = new LlamaContextParams(arena).nCtx(256).nBatch(256);
    var ctx = track(new LlamaContext(arena, model, params));
    var other = track(new LlamaContext(arena, model, params));
    var state = state(ctx, PROMPT);
    assertThatThrownBy(() ->
      state.setDFlash(other, model, SpeculativeConfig.greedy(4))
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("Not a DFlash draft model");
    assertThatThrownBy(() ->
      state.setDFlash(other, model, SpeculativeConfig.ngramGreedy(4, 2))
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("ngram");
  }

  @ParameterizedTest(name = "{2}")
  @MethodSource("drafts")
  void block_draft_is_lossless_under_greedy(
    String targetVar,
    String draftVar,
    DFlashDraft.Kind kind
  ) {
    var paths = paths(targetVar, draftVar);
    var target = track(
      new LlamaModel(arena, paths[0], new LlamaModelParams(arena))
    );
    var draftModel = track(
      new LlamaModel(arena, paths[1], new LlamaModelParams(arena))
    );
    var params = targetParams(1);

    var plainCtx = track(new LlamaContext(arena, target, params));
    String greedy = generate(state(plainCtx, PROMPT));

    var specCtx = track(new LlamaContext(arena, target, params));
    var linkedElsewhere = track(
      new LlamaContext(arena, draftModel, draftParams(plainCtx))
    );
    assertThatThrownBy(() ->
      state(specCtx, PROMPT).setDFlash(
        linkedElsewhere,
        draftModel,
        SpeculativeConfig.greedy(8)
      )
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("ctxOther");

    var draftCtx = track(
      new LlamaContext(arena, draftModel, draftParams(specCtx))
    );
    var specState = state(specCtx, PROMPT).setDFlash(
      draftCtx,
      draftModel,
      SpeculativeConfig.greedy(8)
    );
    assertThat(specState.isDFlash()).isTrue();
    assertThat(specState.getDFlashDraft().kind()).isEqualTo(kind);
    String speculative = generate(specState);

    System.out.println(kind + " greedy : " + greedy);
    System.out.println(kind + " drafted: " + speculative);
    System.out.println(kind + " accept : " + specState.acceptRate());

    assertThat(speculative).contains("ten");
    assertThat(commonPrefix(greedy, speculative)).isGreaterThanOrEqualTo(40);
    assertThat(specState.acceptRate()).isGreaterThan(0.1);
    specState.freeSpeculativeScratch();
  }

  @ParameterizedTest(name = "{2}")
  @MethodSource("drafts")
  void block_draft_runs_sampled_and_adaptive_rounds(
    String targetVar,
    String draftVar,
    DFlashDraft.Kind kind
  ) {
    var paths = paths(targetVar, draftVar);
    var target = track(
      new LlamaModel(arena, paths[0], new LlamaModelParams(arena))
    );
    var draftModel = track(
      new LlamaModel(arena, paths[1], new LlamaModelParams(arena))
    );
    for (SpeculativeConfig config : List.of(
      new SpeculativeConfig(8, 0.7f, 40, 0.95f, 7),
      SpeculativeConfig.greedyAdaptive(8, 1, 0.5f)
    )) {
      var ctx = track(new LlamaContext(arena, target, targetParams(1)));
      var draftCtx = track(
        new LlamaContext(arena, draftModel, draftParams(ctx))
      );
      var state = state(ctx, PROMPT).setDFlash(draftCtx, draftModel, config);
      String text = generate(state);
      System.out.println(
        kind + " " + config + " accept " + state.acceptRate() + ": " + text
      );
      assertThat(text).isNotBlank();
      assertThat(state.getAnswerTokens()).isGreaterThan(10);
      state.freeSpeculativeScratch();
    }
  }

  @ParameterizedTest(name = "{2}")
  @MethodSource("drafts")
  void batch_iterator_drafts_dflash_sequences_together(
    String targetVar,
    String draftVar,
    DFlashDraft.Kind kind
  ) {
    var paths = paths(targetVar, draftVar);
    var target = track(
      new LlamaModel(arena, paths[0], new LlamaModelParams(arena))
    );
    var draftModel = track(
      new LlamaModel(arena, paths[1], new LlamaModelParams(arena))
    );
    var single = track(new LlamaContext(arena, target, targetParams(1)));
    String first = generate(state(single, PROMPT));
    single.clearCache();
    String second = generate(state(single, OTHER_PROMPT));

    var ctx = track(new LlamaContext(arena, target, targetParams(2)));
    var draftCtx = track(
      new LlamaContext(
        arena,
        draftModel,
        draftParams(ctx).nSeqMax(2).nOutputsMax(64)
      )
    );
    var tokenizer = new LlamaTokenizer(new LlamaVocab(target), ctx);
    var a = ConversationState.create(
      arena,
      ctx,
      tokenizer,
      track(new LlamaSampler(arena).greedy()),
      0
    )
      .setMaxTokens(MAX_TOKENS)
      .initialize(PROMPT)
      .setDFlash(draftCtx, draftModel, SpeculativeConfig.greedy(6));
    var b = ConversationState.create(
      arena,
      ctx,
      tokenizer,
      track(new LlamaSampler(arena).greedy()),
      1
    )
      .setMaxTokens(MAX_TOKENS)
      .initialize(OTHER_PROMPT)
      .setDFlash(draftCtx, draftModel, SpeculativeConfig.greedy(6));

    var it = new BatchIterator(arena, ctx).addState(a).addState(b);
    Map<Integer, StringBuilder> out = new HashMap<>();
    it
      .stream()
      .forEach(o ->
        out
          .computeIfAbsent(o.sequenceId(), id -> new StringBuilder())
          .append(o.content())
      );

    System.out.println(kind + " batch 0: " + out.get(0));
    System.out.println(kind + " batch 1: " + out.get(1));
    assertThat(
      commonPrefix(first, out.get(0).toString())
    ).isGreaterThanOrEqualTo(30);
    assertThat(
      commonPrefix(second, out.get(1).toString())
    ).isGreaterThanOrEqualTo(30);
    assertThat(a.acceptRate()).isGreaterThan(0.05);
    assertThat(b.acceptRate()).isGreaterThan(0.05);
  }

  private LlamaContextParams targetParams(int nSeqMax) {
    return new LlamaContextParams(arena)
      .nCtx(1024 * nSeqMax)
      .nBatch(512)
      .nUBatch(512)
      .nSeqMax(nSeqMax);
  }

  private LlamaContextParams draftParams(LlamaContext target) {
    return new LlamaContextParams(arena)
      .nCtx(1024)
      .nBatch(512)
      .nUBatch(512)
      .nOutputsMax(16)
      .ctxOther(target);
  }

  private ConversationState state(LlamaContext ctx, String prompt) {
    var tokenizer = new LlamaTokenizer(new LlamaVocab(ctx.getModel()), ctx);
    return ConversationState.create(
      arena,
      ctx,
      tokenizer,
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .initialize(prompt);
  }

  private static String generate(ConversationState state) {
    return new DefaultLlamaIterator(state)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", String::concat);
  }

  private static int commonPrefix(String a, String b) {
    int n = Math.min(a.length(), b.length());
    int i = 0;
    while (i < n && a.charAt(i) == b.charAt(i)) i++;
    return i;
  }

  private static Path[] paths(String targetVar, String draftVar) {
    var target = System.getenv(targetVar);
    var draft = System.getenv(draftVar);
    assumeTrue(
      target != null &&
        draft != null &&
        Files.exists(Path.of(target)) &&
        Files.exists(Path.of(draft)),
      targetVar + " / " + draftVar + " not set"
    );
    return new Path[] { Path.of(target), Path.of(draft) };
  }
}
