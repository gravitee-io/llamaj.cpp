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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Pure tests for the incremental {@link NgramIndex} used by n-gram (prompt-lookup) drafting — no
 * native library, no model. Covers match selection, most-recent preference, continuation extraction,
 * the kMax cap, and the no-match fallback.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class NgramLookupTest {

  /** Builds an index from {@code history} and returns its proposal for the trailing window. */
  private static int[] propose(int ngram, int kMax, int... history) {
    NgramIndex index = new NgramIndex(ngram);
    for (int token : history) {
      index.append(token);
    }
    return index.propose(kMax);
  }

  @Test
  void no_match_when_history_too_short() {
    // histLen < ngram + 1 → nothing to look up.
    assertThat(propose(2, 4, 1, 2)).isEmpty();
  }

  @Test
  void no_match_returns_empty() {
    // Trailing window [4,5] never occurs earlier.
    assertThat(propose(2, 4, 1, 2, 3, 4, 5)).isEmpty();
  }

  @Test
  void proposes_continuation_after_earlier_match() {
    // Window [1,2]; earlier occurrence at index 0; continuation is history[2..] = [3,1,2].
    assertThat(propose(2, 3, 1, 2, 3, 1, 2)).containsExactly(3, 1, 2);
  }

  @Test
  void prefers_most_recent_earlier_match() {
    // [1,2] occurs at index 1 and (current) index 4; the most recent earlier is index 1 →
    // continuation history[3..] capped to 2 = [8,1].
    assertThat(propose(2, 2, 9, 1, 2, 8, 1, 2)).containsExactly(8, 1);
  }

  @Test
  void caps_proposal_at_kmax() {
    // Match at index 0, continuation [3,4,5,1,2], capped to kMax = 2.
    assertThat(propose(2, 2, 1, 2, 3, 4, 5, 1, 2)).containsExactly(3, 4);
  }

  @Test
  void continuation_bounded_by_history_length() {
    // ngram=1, last token 5 matches at index 1; only one continuation token remains.
    assertThat(propose(1, 5, 5, 5, 5)).containsExactly(5);
  }
}
