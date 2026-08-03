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

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.GenerationState;
import io.gravitee.llama.cpp.StateBounds;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Leaving a channel by more than one marker — the other half of the Harmony
 * (gpt-oss) problem that {@link SharedPrefixMarkerTest} covers for openings.
 *
 * <p>Reasoning reaches the final channel through
 * {@code <|end|><|start|>assistant<|channel|>final<|message|>} when the model
 * answers directly, but through {@code <|call|>} when a tool call intervened.
 * A tool call itself either ends generation — {@code <|call|>} is an EOS token,
 * the normal agent flow — or is followed immediately by the final-channel
 * header. Configure one exit and the other path leaks its header into the
 * answer as visible syntax.
 *
 * <p>The delicate case is a complete match that may still grow: {@code <|call|>}
 * matches while {@code <|call|><|start|>assistant…} may still be arriving.
 * Committing at once makes the longer marker unreachable; waiting without
 * remembering the match leaves the machine stuck inside the tool channel when
 * the longer form never comes.
 *
 * <p>No model or native libraries required.
 *
 * @author GraviteeSource Team
 */
class CloseAlternativesTest {

  private static final String REASON_OPEN = "<|channel|>analysis<|message|>";
  private static final String FINAL_HEADER =
    "<|start|>assistant<|channel|>final<|message|>";
  private static final String TOOL_OPEN = "<|start|>assistant to=functions.";

  private record Run(
    String reasoning,
    String tools,
    String answer,
    GenerationState finalState
  ) {}

  private static Run drive(List<String> pieces) {
    var evaluation = new StateEvaluation();
    evaluation.initialize(
      new StateEvaluation.Config(
        List.of(
          new StateBounds(
            GenerationState.REASONING,
            List.of(REASON_OPEN),
            List.of("<|end|>" + FINAL_HEADER, "<|call|>" + FINAL_HEADER)
          ),
          new StateBounds(
            GenerationState.TOOLS,
            // Both alignments: a close marker sharing the <|end|> prefix buffers
            // it away from the start of the run.
            List.of("<|end|>" + TOOL_OPEN, TOOL_OPEN),
            List.of("<|call|>" + FINAL_HEADER, "<|call|>")
          )
        )
      )
    );

    var reasoning = new StringBuilder();
    var tools = new StringBuilder();
    var answer = new StringBuilder();

    GenerationState state = GenerationState.ANSWER;
    for (String piece : pieces) {
      var emission = evaluation.evaluateToken(state, 0, piece);
      state = emission.state();
      append(reasoning, tools, answer, state, emission.emit());
    }
    var flushed = evaluation.flushPending(state);
    state = flushed.state();
    append(reasoning, tools, answer, state, flushed.emit());

    return new Run(
      reasoning.toString(),
      tools.toString(),
      answer.toString(),
      state
    );
  }

  private static void append(
    StringBuilder reasoning,
    StringBuilder tools,
    StringBuilder answer,
    GenerationState state,
    String text
  ) {
    if (text == null || text.isEmpty()) {
      return;
    }
    switch (state) {
      case REASONING -> reasoning.append(text);
      case TOOLS -> tools.append(text);
      default -> answer.append(text);
    }
  }

  @Test
  void answering_directly_leaves_reasoning_by_the_end_marker() {
    var run = drive(
      List.of(
        "<|channel|>",
        "analysis",
        "<|message|>",
        "Think",
        ".",
        "<|end|>",
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
        "Hello",
        "!"
      )
    );

    assertThat(run.reasoning()).isEqualTo("Think.");
    assertThat(run.answer()).isEqualTo("Hello!");
  }

  @Test
  void a_tool_call_that_ends_generation_still_closes_its_span() {
    // <|call|> is an EOS token, so nothing follows it. The longer alternative
    // never arrives and the span must settle at flush — otherwise the turn ends
    // reported as still inside the tool call, with a stray <|call|> emitted.
    var run = drive(
      List.of(
        "<|channel|>",
        "analysis",
        "<|message|>",
        "Need",
        " a",
        " file",
        "<|end|>",
        "<|start|>",
        "assistant",
        " to=functions.",
        "write",
        "<|channel|>",
        "commentary",
        "<|message|>",
        "{\"p\":1}",
        "<|call|>"
      )
    );

    assertThat(run.tools()).isEqualTo(
      "write<|channel|>commentary<|message|>{\"p\":1}"
    );
    assertThat(run.answer()).doesNotContain("<|call|>");
    assertThat(run.finalState()).isEqualTo(GenerationState.ANSWER);
  }

  @Test
  void a_tool_call_the_model_continues_past_hides_the_final_header() {
    var run = drive(
      List.of(
        "<|channel|>",
        "analysis",
        "<|message|>",
        "Need",
        " a",
        " file",
        "<|end|>",
        "<|start|>",
        "assistant",
        " to=functions.",
        "write",
        "<|channel|>",
        "commentary",
        "<|message|>",
        "{\"p\":1}",
        "<|call|>",
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
        "Created",
        "!"
      )
    );

    assertThat(run.tools()).isEqualTo(
      "write<|channel|>commentary<|message|>{\"p\":1}"
    );
    assertThat(run.answer()).isEqualTo("Created!");
    assertThat(run.answer()).doesNotContain("<|channel|>", "<|start|>");
  }

  @Test
  void the_longest_matching_close_wins() {
    var evaluation = new StateEvaluation();
    evaluation.initialize(
      new StateEvaluation.Config(
        List.of(
          new StateBounds(
            GenerationState.REASONING,
            List.of("<open>"),
            List.of("<close>", "<close>extra")
          )
        )
      )
    );

    var reasoning = new StringBuilder();
    var answer = new StringBuilder();
    GenerationState state = GenerationState.ANSWER;
    for (String piece : List.of("<open>", "body", "<close>", "extra", "tail")) {
      var emission = evaluation.evaluateToken(state, 0, piece);
      state = emission.state();
      append(reasoning, new StringBuilder(), answer, state, emission.emit());
    }
    var flushed = evaluation.flushPending(state);
    append(
      reasoning,
      new StringBuilder(),
      answer,
      flushed.state(),
      flushed.emit()
    );

    // "<close>extra" is the longer match, so "extra" is syntax, not content.
    assertThat(reasoning.toString()).isEqualTo("body");
    assertThat(answer.toString()).isEqualTo("tail");
  }
}
