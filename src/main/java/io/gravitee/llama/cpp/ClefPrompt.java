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

import java.util.Objects;
import java.util.stream.IntStream;

/**
 * A prompt for {@link ClefDecider}: tokens tagged with their role, one {@link DecisionOrder} per
 * token. It is scored into one value per {@link DecisionOrder#OPTION} span, over all the questions
 * of the prompt.
 *
 * @param tokens The prompt tokens
 * @param order  The role of each token
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record ClefPrompt(int[] tokens, DecisionOrder[] order) implements
  DecisionPrompt {
  public ClefPrompt {
    Objects.requireNonNull(tokens, "tokens");
    Objects.requireNonNull(order, "order");
    if (order.length != tokens.length) {
      throw new IllegalArgumentException(
        "order has %d entries for %d tokens".formatted(
          order.length,
          tokens.length
        )
      );
    }
    if (countOptionSpans(order) == 0) {
      throw new IllegalArgumentException("a clef prompt needs an option span");
    }
  }

  /** The number of {@link DecisionOrder#OPTION} spans. */
  @Override
  public int nOptions() {
    return countOptionSpans(order);
  }

  /** Option spans start where an OPTION entry does not follow another. */
  private static int countOptionSpans(DecisionOrder[] order) {
    return (int) IntStream.range(0, order.length)
      .filter(i -> order[i] == DecisionOrder.OPTION)
      .filter(i -> i == 0 || order[i - 1] != DecisionOrder.OPTION)
      .count();
  }
}
