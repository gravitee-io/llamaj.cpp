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

  /**
   * Text-based streaming matching: token ids are irrelevant, only piece text matters.
   * Same initialization as {@link #of} — evaluateToken is always text-aware.
   */
  private static StateEvaluation tokenAware(StateBounds... bounds) {
    return of(bounds);
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
    var eval = of(new StateBounds(TOOLS, (String) null, "<|eom_id|>"));

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

  /* ----- token-sequence matching (evaluateToken) ----- */

  @Test
  void token_aware_single_token_markers_behave_as_before() {
    var eval = tokenAware(new StateBounds(REASONING, "<think>", "</think>"));

    var hello = eval.evaluateToken(ANSWER, 5, "Hello");
    assertThat(hello.state()).isEqualTo(ANSWER);
    assertThat(hello.emit()).isEqualTo("Hello");
    assertThat(hello.emitTokens()).isEqualTo(1);

    var open = eval.evaluateToken(ANSWER, 100, "<think>");
    assertThat(open.state()).isEqualTo(REASONING);
    assertThat(open.emit()).isEmpty(); // marker text suppressed
    assertThat(open.emitTokens()).isEqualTo(1);

    var inner = eval.evaluateToken(REASONING, 6, "pondering");
    assertThat(inner.state()).isEqualTo(REASONING);

    var close = eval.evaluateToken(REASONING, 101, "</think>");
    assertThat(close.state()).isEqualTo(ANSWER);
    assertThat(close.emit()).isEmpty(); // marker text suppressed

    // reasoning occurs at most once
    assertThat(eval.evaluateToken(ANSWER, 100, "<think>").state()).isEqualTo(
      ANSWER
    );
  }

  @Test
  void token_aware_single_token_marker_matches_by_piece_when_id_differs() {
    var eval = tokenAware(new StateBounds(REASONING, "<think>", "</think>"));

    // Model emitted a token with a different id whose text equals the marker.
    var open = eval.evaluateToken(ANSWER, 999, "<think>");
    assertThat(open.state()).isEqualTo(REASONING);
  }

  @Test
  void two_token_open_marker_buffers_then_stamps_reasoning() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    // First marker token: buffered, nothing emitted, still ANSWER.
    var first = eval.evaluateToken(ANSWER, 200, "<|channel>");
    assertThat(first.state()).isEqualTo(ANSWER);
    assertThat(first.emit()).isEmpty();
    assertThat(first.emitTokens()).isZero();
    assertThat(eval.hasPending()).isTrue();

    // Second marker token: full sequence confirmed — whole marker emitted in REASONING.
    var second = eval.evaluateToken(ANSWER, 201, "thought");
    assertThat(second.state()).isEqualTo(REASONING);
    // marker text never leaks into any channel; tokens still counted post-flip
    assertThat(second.emit()).isEmpty();
    assertThat(second.emitTokens()).isEqualTo(2);
    assertThat(eval.hasPending()).isFalse();

    // Reasoning content flows in REASONING.
    assertThat(eval.evaluateToken(REASONING, 7, "hmm").state()).isEqualTo(
      REASONING
    );

    // Two-token close marker: buffer then back to ANSWER with the full close text.
    var closeFirst = eval.evaluateToken(REASONING, 200, "<|channel>");
    assertThat(closeFirst.state()).isEqualTo(REASONING);
    assertThat(closeFirst.emitTokens()).isZero();
    var closeSecond = eval.evaluateToken(REASONING, 202, "end");
    assertThat(closeSecond.state()).isEqualTo(ANSWER);
    assertThat(closeSecond.emit()).isEmpty(); // marker text suppressed
    assertThat(closeSecond.emitTokens()).isEqualTo(2);

    // Reasoning does not re-enter once closed.
    var reOpen = eval.evaluateToken(ANSWER, 200, "<|channel>");
    assertThat(reOpen.state()).isEqualTo(ANSWER);
    assertThat(reOpen.emit()).isEqualTo("<|channel>");
    assertThat(reOpen.emitTokens()).isEqualTo(1);
    assertThat(eval.evaluateToken(ANSWER, 201, "thought").state()).isEqualTo(
      ANSWER
    );
  }

  @Test
  void refuted_prefix_is_emitted_in_current_channel() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    var first = eval.evaluateToken(ANSWER, 200, "<|channel>");
    assertThat(first.emitTokens()).isZero();

    var refuted = eval.evaluateToken(ANSWER, 42, "Hello");
    assertThat(refuted.state()).isEqualTo(ANSWER);
    assertThat(refuted.emit()).isEqualTo("<|channel>Hello");
    assertThat(refuted.emitTokens()).isEqualTo(2);
    assertThat(eval.hasPending()).isFalse();
  }

  @Test
  void refuting_token_can_itself_restart_a_marker_prefix() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    eval.evaluateToken(ANSWER, 200, "<|channel>");
    // Same first marker token again: previous prefix refuted (flushed), new prefix buffered.
    var again = eval.evaluateToken(ANSWER, 200, "<|channel>");
    assertThat(again.state()).isEqualTo(ANSWER);
    assertThat(again.emit()).isEqualTo("<|channel>");
    assertThat(again.emitTokens()).isEqualTo(1);
    assertThat(eval.hasPending()).isTrue();

    var confirmed = eval.evaluateToken(ANSWER, 201, "thought");
    assertThat(confirmed.state()).isEqualTo(REASONING);
    assertThat(confirmed.emit()).isEmpty(); // marker text suppressed
    assertThat(confirmed.emitTokens()).isEqualTo(2);
  }

  @Test
  void flush_pending_returns_buffered_prefix_in_current_channel() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    eval.evaluateToken(ANSWER, 200, "<|channel>");
    var flush = eval.flushPending(ANSWER);
    assertThat(flush.state()).isEqualTo(ANSWER);
    assertThat(flush.emit()).isEqualTo("<|channel>");
    assertThat(flush.emitTokens()).isEqualTo(1);
    assertThat(eval.hasPending()).isFalse();

    // Flushing with nothing pending is a no-op.
    var empty = eval.flushPending(ANSWER);
    assertThat(empty.emitTokens()).isZero();
  }

  @Test
  void two_token_tool_markers_enter_and_exit_tools() {
    var eval = tokenAware(new StateBounds(TOOLS, "<|tool>call", "<|tool>end"));

    assertThat(
      eval.evaluateToken(ANSWER, 210, "<|tool>").emitTokens()
    ).isZero();
    var open = eval.evaluateToken(ANSWER, 211, "call");
    assertThat(open.state()).isEqualTo(TOOLS);
    assertThat(open.emit()).isEmpty(); // marker text suppressed

    assertThat(
      eval.evaluateToken(TOOLS, 8, "{\"name\":\"x\"}").state()
    ).isEqualTo(TOOLS);

    assertThat(eval.evaluateToken(TOOLS, 210, "<|tool>").emitTokens()).isZero();
    var close = eval.evaluateToken(TOOLS, 212, "end");
    assertThat(close.state()).isEqualTo(ANSWER);
    assertThat(close.emit()).isEmpty(); // marker text suppressed

    // tools can reoccur
    eval.evaluateToken(ANSWER, 210, "<|tool>");
    assertThat(eval.evaluateToken(ANSWER, 211, "call").state()).isEqualTo(
      TOOLS
    );
  }

  @Test
  void mixed_single_and_multi_token_markers_coexist() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<think>", "</think>"),
      new StateBounds(TOOLS, "<|tool>call", "<|tool>end")
    );

    assertThat(eval.evaluateToken(ANSWER, 100, "<think>").state()).isEqualTo(
      REASONING
    );
    assertThat(
      eval.evaluateToken(REASONING, 101, "</think>").state()
    ).isEqualTo(ANSWER);

    assertThat(
      eval.evaluateToken(ANSWER, 210, "<|tool>").emitTokens()
    ).isZero();
    assertThat(eval.evaluateToken(ANSWER, 211, "call").state()).isEqualTo(
      TOOLS
    );
  }

  /* ----- tokenization variants of the same marker text ----- */

  @Test
  void fused_single_piece_marker_confirms() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    // The whole marker arrives as ONE piece.
    var fused = eval.evaluateToken(ANSWER, 1, "<|channel>thought");
    assertThat(fused.state()).isEqualTo(REASONING);
    assertThat(fused.emit()).isEmpty();
    assertThat(fused.emitTokens()).isEqualTo(1);
  }

  @Test
  void boundary_spanning_piece_confirms_and_emits_remainder_post_flip() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    assertThat(
      eval.evaluateToken(ANSWER, 1, "<|channel>").emitTokens()
    ).isZero();
    // Piece completes the marker AND carries trailing text: the remainder is the first
    // text of the post-flip channel; marker text stays suppressed.
    var spanning = eval.evaluateToken(ANSWER, 2, "thought\nThe");
    assertThat(spanning.state()).isEqualTo(REASONING);
    assertThat(spanning.emit()).isEqualTo("\nThe");
    assertThat(spanning.emitTokens()).isEqualTo(2);
  }

  @Test
  void single_piece_spanning_single_token_marker_splits() {
    var eval = tokenAware(new StateBounds(REASONING, "<think>", "</think>"));

    var spanning = eval.evaluateToken(ANSWER, 1, "<think>Okay");
    assertThat(spanning.state()).isEqualTo(REASONING);
    assertThat(spanning.emit()).isEqualTo("Okay");
    assertThat(spanning.emitTokens()).isEqualTo(1);
  }

  @Test
  void subword_split_marker_confirms_identically() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    // Same marker text, split at a non-token boundary.
    assertThat(eval.evaluateToken(ANSWER, 1, "<|chan").emitTokens()).isZero();
    var confirmed = eval.evaluateToken(ANSWER, 2, "nel>thought");
    assertThat(confirmed.state()).isEqualTo(REASONING);
    assertThat(confirmed.emit()).isEmpty();
    assertThat(confirmed.emitTokens()).isEqualTo(2);
  }

  @Test
  void three_way_split_marker_confirms() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    assertThat(eval.evaluateToken(ANSWER, 1, "<|ch").emitTokens()).isZero();
    assertThat(eval.evaluateToken(ANSWER, 2, "annel>th").emitTokens()).isZero();
    var confirmed = eval.evaluateToken(ANSWER, 3, "ought");
    assertThat(confirmed.state()).isEqualTo(REASONING);
    assertThat(confirmed.emit()).isEmpty();
    assertThat(confirmed.emitTokens()).isEqualTo(3);
  }

  @Test
  void text_divergence_refutes_and_flushes_current_channel() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    assertThat(eval.evaluateToken(ANSWER, 1, "<|chan").emitTokens()).isZero();
    // Diverges from the marker text mid-way.
    var refuted = eval.evaluateToken(ANSWER, 2, "xyz");
    assertThat(refuted.state()).isEqualTo(ANSWER);
    assertThat(refuted.emit()).isEqualTo("<|chanxyz");
    assertThat(refuted.emitTokens()).isEqualTo(2);
    assertThat(eval.hasPending()).isFalse();
  }

  @Test
  void boundary_spanning_close_marker_returns_remainder_to_answer() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    eval.evaluateToken(ANSWER, 1, "<|channel>thought");
    assertThat(eval.evaluateToken(REASONING, 2, "pondering").state()).isEqualTo(
      REASONING
    );
    assertThat(
      eval.evaluateToken(REASONING, 3, "<|channel>").emitTokens()
    ).isZero();
    var close = eval.evaluateToken(REASONING, 4, "end The answer");
    assertThat(close.state()).isEqualTo(ANSWER);
    assertThat(close.emit()).isEqualTo(" The answer");
    assertThat(close.emitTokens()).isEqualTo(2);
  }

  /* ----- Harmony-style chained channels (gpt-oss) ----- */

  private static final String ANALYSIS_OPEN = "<|channel|>analysis<|message|>";
  private static final String FINAL_FOLD =
    "<|end|><|start|>assistant<|channel|>final<|message|>";
  private static final String TOOL_OPEN =
    "<|end|><|start|>assistant<|channel|>commentary to=functions.";
  private static final String TOOL_CLOSE = "<|call|>";

  private static StateEvaluation harmony() {
    return tokenAware(
      new StateBounds(REASONING, ANALYSIS_OPEN, FINAL_FOLD),
      new StateBounds(TOOLS, TOOL_OPEN, TOOL_CLOSE)
    );
  }

  @Test
  void harmony_analysis_to_final_via_folded_close() {
    var eval = harmony();

    assertThat(eval.evaluateToken(ANSWER, 1, ANALYSIS_OPEN).state()).isEqualTo(
      REASONING
    );
    assertThat(
      eval.evaluateToken(REASONING, 2, "thinking...").state()
    ).isEqualTo(REASONING);

    // Folded close spelled over several pieces sharing a prefix with the tool open.
    assertThat(
      eval.evaluateToken(REASONING, 3, "<|end|>").emitTokens()
    ).isZero();
    assertThat(
      eval.evaluateToken(REASONING, 4, "<|start|>").emitTokens()
    ).isZero();
    assertThat(
      eval.evaluateToken(REASONING, 5, "assistant").emitTokens()
    ).isZero();
    assertThat(
      eval.evaluateToken(REASONING, 6, "<|channel|>").emitTokens()
    ).isZero();
    // "final" disambiguates towards the close; still buffering.
    assertThat(eval.evaluateToken(REASONING, 7, "final").emitTokens()).isZero();
    var closed = eval.evaluateToken(REASONING, 8, "<|message|>");
    assertThat(closed.state()).isEqualTo(ANSWER);
    assertThat(closed.emit()).isEmpty();
    assertThat(closed.emitTokens()).isEqualTo(6);

    assertThat(eval.evaluateToken(ANSWER, 9, "London.").state()).isEqualTo(
      ANSWER
    );
  }

  @Test
  void harmony_analysis_chains_directly_into_tool_call() {
    var eval = harmony();

    eval.evaluateToken(ANSWER, 1, ANALYSIS_OPEN);
    eval.evaluateToken(REASONING, 2, "need the weather tool");

    // Cross-transition: the tool OPEN matches while in REASONING (implicit close).
    assertThat(
      eval
        .evaluateToken(REASONING, 3, "<|end|><|start|>assistant<|channel|>")
        .emitTokens()
    ).isZero();
    var cross = eval.evaluateToken(
      REASONING,
      4,
      "commentary to=functions.get_weather"
    );
    assertThat(cross.state()).isEqualTo(TOOLS);
    // Marker suppressed; boundary-spanning remainder (the function name) lands post-flip.
    assertThat(cross.emit()).isEqualTo("get_weather");
    assertThat(cross.emitTokens()).isEqualTo(2);

    assertThat(
      eval.evaluateToken(TOOLS, 5, " {\"city\":\"Paris\"}").state()
    ).isEqualTo(TOOLS);

    // Tool close, then final text streams as ANSWER.
    var toolClosed = eval.evaluateToken(TOOLS, 6, TOOL_CLOSE);
    assertThat(toolClosed.state()).isEqualTo(ANSWER);
    assertThat(toolClosed.emit()).isEmpty();
    assertThat(eval.evaluateToken(ANSWER, 7, "It is sunny.").state()).isEqualTo(
      ANSWER
    );
  }

  @Test
  void harmony_tool_chains_into_analysis_then_final() {
    var eval = harmony();

    // Straight into a tool call from ANSWER.
    var open = eval.evaluateToken(ANSWER, 1, TOOL_OPEN + "get_capital");
    assertThat(open.state()).isEqualTo(TOOLS);
    assertThat(open.emit()).isEqualTo("get_capital");

    // Cross-transition TOOLS → REASONING on the analysis open.
    var cross = eval.evaluateToken(TOOLS, 2, ANALYSIS_OPEN);
    assertThat(cross.state()).isEqualTo(REASONING);
    assertThat(cross.emit()).isEmpty();
    assertThat(cross.emitTokens()).isEqualTo(1);

    assertThat(eval.evaluateToken(REASONING, 3, "got it").state()).isEqualTo(
      REASONING
    );

    // Folded close (fused in one piece, with remainder) ends reasoning into ANSWER.
    var closed = eval.evaluateToken(REASONING, 4, FINAL_FOLD + "Paris");
    assertThat(closed.state()).isEqualTo(ANSWER);
    assertThat(closed.emit()).isEqualTo("Paris");
    assertThat(closed.emitTokens()).isEqualTo(1);
  }

  @Test
  void shared_prefix_longest_match_prefers_longer_candidate() {
    var eval = tokenAware(
      new StateBounds(REASONING, "<r>", "<|end|>"),
      new StateBounds(TOOLS, "<|end|><|tool|>", "<|call|>")
    );

    eval.evaluateToken(ANSWER, 1, "<r>");
    // The fused piece covers BOTH the reasoning close "<|end|>" and the tool open
    // "<|end|><|tool|>" — the longest candidate wins, transitioning directly to TOOLS.
    var cross = eval.evaluateToken(REASONING, 2, "<|end|><|tool|>{");
    assertThat(cross.state()).isEqualTo(TOOLS);
    assertThat(cross.emit()).isEqualTo("{");
    assertThat(cross.emitTokens()).isEqualTo(1);
  }

  @Test
  void cross_transition_marks_reasoning_occurred() {
    var eval = harmony();

    eval.evaluateToken(ANSWER, 1, ANALYSIS_OPEN);
    eval.evaluateToken(REASONING, 2, TOOL_OPEN + "f");
    eval.evaluateToken(TOOLS, 3, TOOL_CLOSE);

    // Reasoning implicitly closed by the cross-transition: it must not re-enter.
    var reOpen = eval.evaluateToken(ANSWER, 4, ANALYSIS_OPEN);
    assertThat(reOpen.state()).isEqualTo(ANSWER);
    assertThat(reOpen.emit()).isEqualTo(ANALYSIS_OPEN);
  }

  @Test
  void single_state_config_ignores_cross_transitions() {
    var eval = tokenAware(new StateBounds(REASONING, "<think>", "</think>"));

    eval.evaluateToken(ANSWER, 1, "<think>");
    // With only one state configured there are no cross candidates: unrelated tag-like
    // text streams through REASONING untouched, close still works — exactly as before.
    var inner = eval.evaluateToken(REASONING, 2, "<tool_call>");
    assertThat(inner.state()).isEqualTo(REASONING);
    assertThat(inner.emit()).isEqualTo("<tool_call>");
    assertThat(eval.evaluateToken(REASONING, 3, "</think>").state()).isEqualTo(
      ANSWER
    );
  }

  /* ----- input-side: prompt ending inside an unfinished span ----- */

  @Test
  void initial_state_enters_reasoning_when_prompt_ends_inside_open_span() {
    var eval = of(new StateBounds(REASONING, "<think>", "</think>"));

    assertThat(
      eval.initialState(
        "<|im_start|>assistant\n<think>partial reasoning so far"
      )
    ).isEqualTo(REASONING);
  }

  @Test
  void initial_state_is_answer_when_last_span_is_closed() {
    var eval = of(new StateBounds(REASONING, "<think>", "</think>"));

    assertThat(eval.initialState("<think>done</think>The answer is")).isEqualTo(
      ANSWER
    );
  }

  @Test
  void initial_state_detects_multi_token_marker_span_textually() {
    var eval = of(
      new StateBounds(REASONING, "<|channel>thought", "<|channel>end")
    );

    assertThat(
      eval.initialState("prompt\n<|channel>thought partial")
    ).isEqualTo(REASONING);
    assertThat(
      eval.initialState("prompt\n<|channel>thought done<|channel>end answer")
    ).isEqualTo(ANSWER);
  }
}
