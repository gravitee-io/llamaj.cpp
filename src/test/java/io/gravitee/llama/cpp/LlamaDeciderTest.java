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
import static org.assertj.core.data.Offset.offset;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link LlamaDecider} and {@link DecisionType}.
 * <p>
 * The decision-model cases run only when {@code LLAMAJ_DECISION_MODEL} points at a decision GGUF
 * (converted with llama.cpp's converter); each case also skips unless the model is of its family.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class LlamaDeciderTest extends LlamaCppTest {

  // Split so prettier-java does not lex "|>" as a token.
  private static final String IM_START = "<|" + "im_start|>";
  private static final String IM_END = "<|" + "im_end|>";

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
  }

  @Test
  void ordinary_model_is_not_a_decision_model() {
    var model = track(
      new LlamaModel(
        arena,
        getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
        new LlamaModelParams(arena)
      )
    );

    assertThat(DecisionType.of(arena, model)).isEmpty();
    assertThatThrownBy(() ->
      LlamaDecider.create(arena, model, LlamaDecider.Options.defaults())
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("not a decision model");
  }

  @Test
  void custom_decider_scores_any_model_on_chosen_codes() {
    var model = track(
      new LlamaModel(
        arena,
        getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
        new LlamaModelParams(arena)
      )
    );
    List<String> codes = List.of("positive", "negative", "neutral");
    try (
      var decider = LabelDecider.custom(
        arena,
        model,
        LlamaDecider.Options.defaults(),
        codes
      )
    ) {
      assertThat(decider.type()).isEqualTo(DecisionType.CUSTOM);
      assertThat(decider.labels())
        .extracting(LabelDecider.Label::code)
        .containsExactlyElementsOf(codes);

      // The label is one token right after a {"sentiment": " prefill.
      var prompt = chat(
        "Classify the sentiment as positive, negative or neutral.",
        "The delivery was late and the box was crushed.",
        "{\"sentiment\": \""
      );
      var scores = decider.score(
        List.of(TextPrompt.of(prompt, 3), TextPrompt.of(prompt, 2))
      );
      assertThat(scores.get(0)).hasSize(3);
      assertThat(scores.get(1)).hasSize(2);
      assertFinite(scores.get(0));
      // Fewer options read the same leading labels.
      assertThat(scores.get(1)[0]).isCloseTo(scores.get(0)[0], offset(0.05f));
    }
  }

  @Test
  void untyped_score_rejects_a_prompt_of_another_kind() {
    var model = track(
      new LlamaModel(
        arena,
        getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
        new LlamaModelParams(arena)
      )
    );
    try (
      LlamaDecider decider = LabelDecider.custom(
        arena,
        model,
        LlamaDecider.Options.defaults(),
        List.of("positive", "negative")
      )
    ) {
      assertThatThrownBy(() ->
        decider.score(new LayaPrompt(new int[] { 1 }, QuestionType.CHOICE, 1))
      )
        .isInstanceOf(LlamaException.class)
        .hasMessageContaining("TextPrompt");
    }
  }

  @Test
  void family_factory_rejects_a_model_of_another_family() {
    var model = track(
      new LlamaModel(
        arena,
        getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
        new LlamaModelParams(arena)
      )
    );
    assertThatThrownBy(() ->
      LayaDecider.create(arena, model, LlamaDecider.Options.defaults())
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("not a decision model");
  }

  @Test
  void custom_decider_rejects_codes_that_are_not_one_distinct_token() {
    var model = track(
      new LlamaModel(
        arena,
        getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
        new LlamaModelParams(arena)
      )
    );
    var options = LlamaDecider.Options.defaults();
    assertThatThrownBy(() ->
      LabelDecider.custom(
        arena,
        model,
        options,
        List.of("positive", "antidisestablishmentarianism")
      )
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("must be one");
    assertThatThrownBy(() ->
      LabelDecider.custom(
        arena,
        model,
        options,
        List.of("positive", "positive")
      )
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("repeats");
  }

  @Test
  void decision_type_parses_known_names() {
    assertThat(DecisionType.fromMetaName("nimble")).isEqualTo(
      DecisionType.NIMBLE
    );
    assertThat(DecisionType.fromMetaName("clef")).isEqualTo(DecisionType.CLEF);
    assertThatThrownBy(() -> DecisionType.fromMetaName("future"))
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("future");
  }

  @Test
  void prompts_validate_their_shape() {
    assertThatThrownBy(() ->
      new TextPrompt("x", new int[] { 1 }, 1)
    ).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> TextPrompt.of("x", 0)).isInstanceOf(
      IllegalArgumentException.class
    );
    assertThatThrownBy(() ->
      new LayaPrompt(new int[] { 1 }, null, 1)
    ).isInstanceOf(NullPointerException.class);

    DecisionOrder N = DecisionOrder.NONE;
    DecisionOrder Q = DecisionOrder.QUESTION_CHOICE;
    DecisionOrder O = DecisionOrder.OPTION;
    var clef = new ClefPrompt(
      new int[] { 1, 2, 3, 4, 5, 6, 7 },
      new DecisionOrder[] { Q, N, O, O, N, O, N }
    );
    assertThat(clef.nOptions()).isEqualTo(2);
    assertThatThrownBy(() ->
      new ClefPrompt(new int[] { 1, 2 }, new DecisionOrder[] { Q, N })
    ).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void factory_returns_the_decider_of_the_family() {
    LlamaModel model = decisionModel(EnumSet.allOf(DecisionType.class));
    try (
      var decider = LlamaDecider.create(
        arena,
        model,
        LlamaDecider.Options.defaults()
      )
    ) {
      Class<?> expected = switch (decider.type()) {
        case OPENJEV, LEV, NIMBLE, CUSTOM -> LabelDecider.class;
        case KEV -> KevDecider.class;
        case LAYA -> LayaDecider.class;
        case CLEF -> ClefDecider.class;
      };
      assertThat(decider).isInstanceOf(expected);
    }
  }

  @Test
  void shared_prefix_scores_match_isolated_scores() {
    LlamaModel model = decisionModel(LABEL_TYPES);
    List<TextPrompt> prompts = List.of(
      TextPrompt.of(prompt("Requested field: \"fruit\""), 3),
      TextPrompt.of(prompt("Requested field: \"colour\""), 3),
      TextPrompt.of(prompt("Requested field: \"size\""), 2)
    );

    // LlamaDecider is not MemorySegmentAware, so track() would not free it: close it explicitly.
    List<float[]> withSharing;
    try (
      var shared = LabelDecider.create(
        arena,
        model,
        LlamaDecider.Options.defaults()
      )
    ) {
      assertThat(shared.labels()).isNotEmpty();
      assertThat(shared.labels().get(0).code()).isEqualTo("A");
      withSharing = shared.score(prompts);
    }

    List<float[]> withoutSharing;
    try (
      var isolated = LlamaDecider.create(
        arena,
        model,
        LlamaDecider.Options.defaults().withNSeqMax(1)
      )
    ) {
      withoutSharing = isolated.score(prompts);
    }

    assertThat(withSharing).hasSize(3);
    IntStream.range(0, prompts.size()).forEach(i -> {
      assertThat(withSharing.get(i)).hasSize(prompts.get(i).nOptions());
      assertThat(withSharing.get(i)).containsExactly(
        withoutSharing.get(i),
        offset(0.05f)
      );
    });
  }

  @Test
  void too_many_options_are_rejected() {
    try (
      var decider = LabelDecider.create(
        arena,
        decisionModel(LABEL_TYPES),
        LlamaDecider.Options.defaults()
      )
    ) {
      int max = decider.labels().size();
      assertThatThrownBy(() ->
        decider.score(TextPrompt.of(prompt("x"), max + 1))
      ).isInstanceOf(LlamaException.class);
    }
  }

  @Test
  void kev_scores_one_value_per_marker() {
    try (
      var decider = LlamaDecider.create(
        arena,
        decisionModel(Set.of(DecisionType.KEV)),
        LlamaDecider.Options.defaults()
      )
    ) {
      String box = "<|" + "box_end|>";
      float[] scores = decider.score(
        TextPrompt.of(
          "Is the apple red?\nyes" +
            box +
            "\nno" +
            box +
            "\nA small red apple.",
          2
        )
      );
      assertThat(scores).hasSize(2);
      assertFinite(scores);
    }
  }

  @Test
  void laya_scores_one_value_per_marker() {
    try (
      var decider = LayaDecider.create(
        arena,
        decisionModel(Set.of(DecisionType.LAYA)),
        LlamaDecider.Options.defaults()
      )
    ) {
      // [question] [sep] ([marker] option)* [sep] state [sep]
      int[] sep = { decider.sepToken() };
      int[] mask = { decider.markerToken() };
      int[] tokens = Stream.of(
        decider.tokenize("Is the apple red?", false),
        sep,
        mask,
        decider.tokenize("yes", false),
        mask,
        decider.tokenize("no", false),
        sep,
        decider.tokenize("A small red apple.", false),
        sep
      )
        .flatMapToInt(Arrays::stream)
        .toArray();

      float[] scores = decider.score(
        new LayaPrompt(tokens, QuestionType.CHOICE, 2)
      );
      assertThat(scores).hasSize(2);
      assertFinite(scores);
    }
  }

  @Test
  void clef_scores_one_value_per_option_span() {
    assumeTrue(
      LlamaExt.decisionAvailable(),
      LlamaExt.decisionResolutionReport()
    );
    try (
      var decider = LlamaDecider.create(
        arena,
        decisionModel(Set.of(DecisionType.CLEF)),
        LlamaDecider.Options.defaults()
      )
    ) {
      record Span(int[] tokens, DecisionOrder role) {}
      var spans = Stream.of(
        Map.entry("A small red apple.\n", DecisionOrder.NONE),
        Map.entry("Which colour?", DecisionOrder.QUESTION_CHOICE),
        Map.entry("\n", DecisionOrder.NONE),
        Map.entry("red", DecisionOrder.OPTION),
        Map.entry("\n", DecisionOrder.NONE),
        Map.entry("green", DecisionOrder.OPTION),
        Map.entry("\n", DecisionOrder.NONE)
      )
        .map(e -> new Span(decider.tokenize(e.getKey(), false), e.getValue()))
        .toList();
      int[] tokens = spans
        .stream()
        .flatMapToInt(span -> Arrays.stream(span.tokens()))
        .toArray();
      var order = spans
        .stream()
        .flatMap(span ->
          Collections.nCopies(span.tokens().length, span.role()).stream()
        )
        .toArray(DecisionOrder[]::new);

      float[] scores = decider.score(new ClefPrompt(tokens, order));
      assertThat(scores).hasSize(2);
    }
  }

  private static final Set<DecisionType> LABEL_TYPES = EnumSet.of(
    DecisionType.OPENJEV,
    DecisionType.LEV,
    DecisionType.NIMBLE
  );

  /** Loads LLAMAJ_DECISION_MODEL, skipping the test unless it is set and of one of the families. */
  private LlamaModel decisionModel(Set<DecisionType> families) {
    var path = System.getenv("LLAMAJ_DECISION_MODEL");
    assumeTrue(
      path != null && Files.exists(Path.of(path)),
      "LLAMAJ_DECISION_MODEL not set"
    );
    var model = track(
      new LlamaModel(arena, Path.of(path), new LlamaModelParams(arena))
    );
    var type = DecisionType.of(arena, model).orElse(null);
    assumeTrue(
      families.contains(type),
      "decision model is %s, not one of %s".formatted(type, families)
    );
    return model;
  }

  private static void assertFinite(float[] scores) {
    IntStream.range(0, scores.length).forEach(i ->
      assertThat(Float.isFinite(scores[i])).as("score %d", i).isTrue()
    );
  }

  private static String prompt(String tail) {
    return chat(
      "Answer with the code of one choice.",
      "{\"context\": \"A small red apple.\", \"schema\": []}\n\n" + tail,
      ""
    );
  }

  /** A ChatML exchange with thinking disabled, the assistant turn prefilled. */
  private static String chat(String system, String user, String prefill) {
    return String.join(
      "\n",
      IM_START + "system\n" + system + IM_END,
      IM_START + "user\n" + user + IM_END,
      IM_START + "assistant\n<think>\n\n</think>\n\n" + prefill
    );
  }
}
