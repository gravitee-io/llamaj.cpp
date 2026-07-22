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
package io.gravitee.llama.cpp.modules;

import static io.gravitee.llama.cpp.GenerationState.ANSWER;
import static io.gravitee.llama.cpp.GenerationState.REASONING;
import static io.gravitee.llama.cpp.GenerationState.TOOLS;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.GenerationState;
import io.gravitee.llama.cpp.StateBounds;
import java.util.List;
import org.junit.jupiter.api.Test;

class StateEvaluationTest {

  private static StateEvaluation of(StateBounds... bounds) {
    var eval = new StateEvaluation();
    eval.initialize(new StateEvaluation.Config(List.of(bounds)));
    return eval;
  }

  private static GenerationState step(
    StateEvaluation eval,
    GenerationState current,
    String piece
  ) {
    return eval.evaluate(new StateEvaluation.Context(current, piece));
  }

  @Test
  void uninitialized_always_answers() {
    var eval = new StateEvaluation();
    assertThat(eval.isInitialized()).isFalse();
    assertThat(step(eval, ANSWER, "<think>")).isEqualTo(ANSWER);
  }

  @Test
  void null_current_state_resolves_to_answer() {
    var eval = of(new StateBounds(REASONING, "<think>", "</think>"));
    assertThat(step(eval, null, "anything")).isEqualTo(ANSWER);
  }

  @Test
  void tagged_reasoning_enters_and_exits_on_delimiters() {
    var eval = of(new StateBounds(REASONING, "<think>", "</think>"));

    assertThat(step(eval, ANSWER, "Hello")).isEqualTo(ANSWER);
    assertThat(step(eval, ANSWER, "<think>")).isEqualTo(REASONING);
    assertThat(step(eval, REASONING, "pondering")).isEqualTo(REASONING);
    assertThat(step(eval, REASONING, "</think>")).isEqualTo(ANSWER);
  }

  @Test
  void reasoning_occurs_at_most_once() {
    var eval = of(new StateBounds(REASONING, "<think>", "</think>"));

    step(eval, ANSWER, "<think>");
    step(eval, REASONING, "</think>");

    assertThat(step(eval, ANSWER, "<think>")).isEqualTo(ANSWER);
  }

  @Test
  void tagged_tools_can_reoccur() {
    var eval = of(new StateBounds(TOOLS, "<tool_call>", "</tool_call>"));

    assertThat(step(eval, ANSWER, "<tool_call>")).isEqualTo(TOOLS);
    assertThat(step(eval, TOOLS, "{\"name\":\"x\"}")).isEqualTo(TOOLS);
    assertThat(step(eval, TOOLS, "</tool_call>")).isEqualTo(ANSWER);

    assertThat(step(eval, ANSWER, "<tool_call>")).isEqualTo(TOOLS);
  }

  @Test
  void non_delimiter_pieces_stay_in_answer() {
    var eval = of(
      new StateBounds(REASONING, "<think>", "</think>"),
      new StateBounds(TOOLS, "<tool_call>", "</tool_call>")
    );
    assertThat(step(eval, ANSWER, "just")).isEqualTo(ANSWER);
    assertThat(step(eval, ANSWER, "text")).isEqualTo(ANSWER);
  }

  @Test
  void blank_start_bounds_never_match_a_generated_piece() {
    var eval = of(new StateBounds(REASONING, "", "</think>"));

    assertThat(step(eval, ANSWER, "first")).isEqualTo(ANSWER);
    assertThat(step(eval, ANSWER, "more")).isEqualTo(ANSWER);
  }

  @Test
  void null_start_bounds_never_match_a_generated_piece() {
    var eval = of(new StateBounds(TOOLS, null, "<|eom_id|>"));

    assertThat(step(eval, ANSWER, "{\"name\"")).isEqualTo(ANSWER);
  }

  @Test
  void initial_state_is_answer_when_prompt_lacks_open_tag() {
    var eval = of(new StateBounds(REASONING, "<think>", "</think>"));

    assertThat(eval.initialState("<|im_start|>assistant\n")).isEqualTo(ANSWER);
  }

  @Test
  void initial_state_enters_reasoning_when_prompt_ends_with_open_tag() {
    var eval = of(new StateBounds(REASONING, "<think>", "</think>"));

    assertThat(eval.initialState("<|im_start|>assistant\n<think>")).isEqualTo(
      REASONING
    );
  }

  @Test
  void initial_state_ignores_trailing_whitespace_after_open_tag() {
    var eval = of(new StateBounds(REASONING, "<think>", "</think>"));

    assertThat(
      eval.initialState("<|im_start|>assistant\n<think>\n\n")
    ).isEqualTo(REASONING);
  }

  @Test
  void initial_state_ignores_blank_start_bounds() {
    var eval = of(new StateBounds(REASONING, "", "</think>"));

    assertThat(eval.initialState("any prompt at all")).isEqualTo(ANSWER);
  }

  @Test
  void initial_state_is_answer_when_uninitialized_or_null_prompt() {
    var uninitialized = new StateEvaluation();
    assertThat(uninitialized.initialState("prompt<think>")).isEqualTo(ANSWER);

    var eval = of(new StateBounds(REASONING, "<think>", "</think>"));
    assertThat(eval.initialState(null)).isEqualTo(ANSWER);
  }

  @Test
  void seeded_reasoning_exits_on_end_tag_and_does_not_reenter() {
    var eval = of(new StateBounds(REASONING, "<think>", "</think>"));
    var state = eval.initialState("prompt ending with <think>\n");
    assertThat(state).isEqualTo(REASONING);

    assertThat(step(eval, state, "pondering")).isEqualTo(REASONING);
    assertThat(step(eval, REASONING, "</think>")).isEqualTo(ANSWER);
    assertThat(step(eval, ANSWER, "<think>")).isEqualTo(ANSWER);
  }
}
