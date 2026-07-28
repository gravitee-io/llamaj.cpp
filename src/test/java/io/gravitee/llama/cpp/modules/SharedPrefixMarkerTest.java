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
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Chained grammars whose markers share a leading run — the Harmony (gpt-oss) case.
 *
 * <p>Reasoning-close and tool-open agree for 34 characters:
 * <pre>
 *   reasoning_close: &lt;|end|&gt;&lt;|start|&gt;assistant&lt;|channel|&gt;final&lt;|message|&gt;
 *   tool_open:       &lt;|end|&gt;&lt;|start|&gt;assistant&lt;|channel|&gt;commentary to=functions.
 *                    └──────────── identical ────────────┘
 * </pre>
 *
 * <p>While that shared run is buffered the machine cannot yet know which marker it is looking at.
 * What matters downstream is that a run which resolves to the reasoning-close emits <b>no tool
 * tokens</b> — a caller that reports a tool call on a zero-token span invites the response layer to
 * go looking for a call in the plain answer and invent one from the first word it finds.
 *
 * <p>No model or native libraries required.
 *
 * @author GraviteeSource Team
 */
class SharedPrefixMarkerTest {

  private static final String REASON_OPEN = "<|channel|>analysis<|message|>";
  private static final String REASON_CLOSE =
    "<|end|><|start|>assistant<|channel|>final<|message|>";
  private static final String TOOL_OPEN =
    "<|end|><|start|>assistant<|channel|>commentary to=functions.";
  /** gpt-oss is told to use commentary, but that is a system-prompt request, not a grammar. */
  private static final String TOOL_OPEN_ANALYSIS =
    "<|end|><|start|>assistant<|channel|>analysis to=functions.";
  private static final String TOOL_CLOSE = "<|call|>";

  /** Mirrors TokenTracking: accumulates emitted tokens per channel. */
  private record Run(
    Map<GenerationState, Integer> tokens,
    String answer,
    String reasoning
  ) {}

  /**
   * Drives the real state machine over {@code pieces}, one token each, exactly as
   * {@code LlamaIterator.processSampledToken} does.
   */
  private static Run drive(List<String> pieces) {
    var evaluation = new StateEvaluation();
    evaluation.initialize(
      new StateEvaluation.Config(
        List.of(
          new StateBounds(GenerationState.REASONING, REASON_OPEN, REASON_CLOSE),
          new StateBounds(
            GenerationState.TOOLS,
            List.of(TOOL_OPEN, TOOL_OPEN_ANALYSIS),
            TOOL_CLOSE
          )
        )
      )
    );

    Map<GenerationState, Integer> tokens = new EnumMap<>(GenerationState.class);
    for (GenerationState s : GenerationState.values()) {
      tokens.put(s, 0);
    }
    var answer = new StringBuilder();
    var reasoning = new StringBuilder();

    GenerationState state = GenerationState.ANSWER;
    for (String piece : pieces) {
      var emission = evaluation.evaluateToken(state, 0, piece);
      state = emission.state();
      tokens.merge(state, emission.emitTokens(), Integer::sum);
      if (!emission.emit().isEmpty()) {
        (state == GenerationState.REASONING ? reasoning : answer).append(
          emission.emit()
        );
      }
    }
    var flushed = evaluation.flushPending(state);
    if (!flushed.emit().isEmpty()) {
      tokens.merge(flushed.state(), flushed.emitTokens(), Integer::sum);
      (flushed.state() == GenerationState.REASONING
          ? reasoning
          : answer).append(flushed.emit());
    }
    return new Run(tokens, answer.toString(), reasoning.toString());
  }

  /** Splits text into small pieces so markers straddle token boundaries, as they really do. */
  private static List<String> tokenize(String... chunks) {
    List<String> out = new ArrayList<>();
    for (String chunk : chunks) {
      for (int i = 0; i < chunk.length(); i += 3) {
        out.add(chunk.substring(i, Math.min(i + 3, chunk.length())));
      }
    }
    return out;
  }

  /**
   * The reported bug: the model reasons, decides to ASK the user rather than call a tool, and
   * ends its analysis into the final channel. Nothing may be attributed to TOOLS.
   */
  @Test
  void reasoning_that_ends_into_the_final_channel_emits_no_tool_tokens() {
    var run = drive(
      tokenize(
        REASON_OPEN,
        "User wants to send an email. We don't have Jamie's address. Let's ask.",
        REASON_CLOSE,
        "Could you give me Jamie's email address?"
      )
    );

    assertThat(run.tokens().get(GenerationState.TOOLS))
      .as("a refuted shared prefix must not be billed to TOOLS")
      .isZero();
    assertThat(run.answer()).isEqualTo(
      "Could you give me Jamie's email address?"
    );
    assertThat(run.reasoning()).contains("Let's ask.");
  }

  /** The genuine tool call still registers — the guard must not suppress real spans. */
  @Test
  void reasoning_that_chains_into_a_tool_call_does_emit_tool_tokens() {
    var run = drive(
      tokenize(
        REASON_OPEN,
        "Jamie's address is known. Call send_email.",
        TOOL_OPEN,
        "send_email<|constrain|>json<|message|>{\"to\":\"jamie@example.com\"}",
        TOOL_CLOSE
      )
    );

    assertThat(run.tokens().get(GenerationState.TOOLS))
      .as("a resolved tool span must be billed to TOOLS")
      .isPositive();
  }

  /**
   * The variant that leaked in practice: the model opens the tool call on the ANALYSIS channel
   * instead of commentary. With only the commentary marker configured the run refutes against
   * reasoning-close, and the entire call — markers included — is flushed into the reasoning
   * channel as text; the reasoning never closes.
   */
  @Test
  void a_tool_call_on_the_analysis_channel_is_recognised_too() {
    var run = drive(
      tokenize(
        REASON_OPEN,
        "Need to inspect the file.",
        TOOL_OPEN_ANALYSIS,
        "read_file<|constrain|>json<|message|>{\"path\":\"/tmp/x\"}",
        TOOL_CLOSE
      )
    );

    assertThat(run.tokens().get(GenerationState.TOOLS))
      .as("the analysis-channel variant must open a tool span")
      .isPositive();
    assertThat(run.reasoning())
      .as("marker text must not leak into reasoning")
      .doesNotContain("to=functions.");
  }

  /** A tool call with no marker prefix ambiguity behaves the same. */
  @Test
  void plain_answer_never_touches_tools() {
    var run = drive(tokenize("The capital of France is Paris."));

    assertThat(run.tokens().get(GenerationState.TOOLS)).isZero();
    assertThat(run.answer()).isEqualTo("The capital of France is Paris.");
  }
}
