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

import java.util.List;

/**
 * The marker text opening and closing one generation channel.
 *
 * <p>{@code starts} is a list because a dialect may open a channel more than one way — Harmony
 * emits tool calls on both the {@code commentary} and {@code analysis} channels. Configure every
 * variant: an unmatched one refutes against the close marker and leaks the span as text.
 *
 * <p>Markers are matched only at the start of a run, so each alternative must be complete from
 * that boundary; a shared substring will not match.
 *
 * @param state  the channel these bounds delimit
 * @param starts opening markers; longest match wins
 * @param end    the closing marker
 */
public record StateBounds(
  GenerationState state,
  List<String> starts,
  List<String> ends
) {
  public StateBounds {
    starts = starts == null ? List.of() : List.copyOf(starts);
    ends = ends == null ? List.of() : List.copyOf(ends);
  }

  /** Single-marker form. */
  public StateBounds(GenerationState state, String start, String end) {
    this(
      state,
      start == null ? List.of() : List.of(start),
      end == null ? List.of() : List.of(end)
    );
  }

  /** Many openings, one closing. */
  public StateBounds(GenerationState state, List<String> starts, String end) {
    this(state, starts, end == null ? List.of() : List.of(end));
  }

  /**
   * The primary closing marker, or {@code null} if none.
   *
   * <p>A channel may be left more than one way — Harmony reaches the final
   * channel after {@code <|end|>} when the model answers directly and after
   * {@code <|call|>} when a tool call intervened — so callers that must consider
   * every exit use {@link #ends()}.
   */
  public String end() {
    return ends.isEmpty() ? null : ends.getFirst();
  }

  /** The primary opening marker, or {@code null} if none. */
  public String start() {
    return starts.isEmpty() ? null : starts.getFirst();
  }
}
