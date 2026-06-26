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
import static org.assertj.core.api.Assertions.within;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/**
 * Pure tests for the adaptive-greedy confidence read ({@link Speculation#draftGreedyConfident}) —
 * no native library, no model. The rejection-sampling accept/residual and the temp/top-k/top-p
 * sampling are exercised by the (native) integration tests instead.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class SpeculationTest {

  private static Speculation spec(Arena arena) {
    return new Speculation(
      arena,
      8,
      new SpeculativeConfig(2, 0.8f, 0, 1.0f, 7)
    );
  }

  @Test
  void draft_greedy_confident_returns_argmax_and_top_softmax_prob() {
    try (Arena arena = Arena.ofConfined()) {
      Speculation s = spec(arena);
      int nVocab = 4;
      MemorySegment logits = arena.allocate(ValueLayout.JAVA_FLOAT, nVocab);
      float[] vals = { 1.0f, 3.0f, 2.0f, 0.0f }; // argmax at id 1
      for (int i = 0; i < nVocab; i++) {
        logits.setAtIndex(ValueLayout.JAVA_FLOAT, i, vals[i]);
      }
      float[] probOut = new float[1];

      int argmax = s.draftGreedyConfident(logits, nVocab, probOut);

      assertThat(argmax).isEqualTo(1);
      // top prob = 1 / Σ exp(logit - maxLogit) = 1 / (e^-2 + 1 + e^-1 + e^-3) ≈ 0.6439
      assertThat(probOut[0]).isCloseTo(0.6439f, within(1e-3f));
    }
  }

  @Test
  void draft_greedy_confident_breaks_ties_by_lowest_id() {
    try (Arena arena = Arena.ofConfined()) {
      Speculation s = spec(arena);
      int nVocab = 3;
      MemorySegment logits = arena.allocate(ValueLayout.JAVA_FLOAT, nVocab);
      logits.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 2.0f);
      logits.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 2.0f); // tie with id 0
      logits.setAtIndex(ValueLayout.JAVA_FLOAT, 2, 1.0f);
      float[] probOut = new float[1];

      // strict '>' keeps the first (lowest-id) max, matching the native greedy sampler.
      assertThat(s.draftGreedyConfident(logits, nVocab, probOut)).isEqualTo(0);
    }
  }
}
