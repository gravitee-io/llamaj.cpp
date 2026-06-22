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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/**
 * Pure-FFM tests for {@link LlamaTokenDataArray} — exercises the candidate-buffer struct
 * layout (fill / size / selected reset / probability reads) with only an {@link Arena}, no
 * native library and no model. Guards against layout regressions in the diffusion sampling
 * path without the gated GGUF download.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class LlamaTokenDataArrayTest {

  @Test
  void fill_sets_size_resets_selection_and_zeroes_probabilities() {
    try (Arena arena = Arena.ofConfined()) {
      int nVocab = 4;
      MemorySegment logits = arena.allocate(ValueLayout.JAVA_FLOAT, nVocab);
      for (int i = 0; i < nVocab; i++) {
        logits.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 1.0f);
      }

      var candidates = new LlamaTokenDataArray(arena, nVocab);
      candidates.fill(logits, 0);

      assertThat(candidates.size()).isEqualTo(nVocab);
      assertThat(candidates.nVocab()).isEqualTo(nVocab);
      // fill() resets selection to -1 and probabilities to 0 (set by the sampler later).
      assertThat(candidates.selectedIndex()).isEqualTo(-1L);
      for (int i = 0; i < nVocab; i++) {
        assertThat(candidates.probabilityAt(i)).isZero();
      }
    }
  }

  @Test
  void selectedId_before_apply_fails_clearly() {
    try (Arena arena = Arena.ofConfined()) {
      int nVocab = 3;
      MemorySegment logits = arena.allocate(ValueLayout.JAVA_FLOAT, nVocab);
      var candidates = new LlamaTokenDataArray(arena, nVocab);
      candidates.fill(logits, 0);

      // No sampler has run, so nothing is selected yet.
      assertThatThrownBy(candidates::selectedId)
        .isInstanceOf(LlamaException.class);
    }
  }

  @Test
  void fill_reads_the_requested_row_offset() {
    try (Arena arena = Arena.ofConfined()) {
      int nVocab = 2;
      // Two rows of logits laid out contiguously.
      MemorySegment logits = arena.allocate(ValueLayout.JAVA_FLOAT, 2L * nVocab);
      for (int i = 0; i < 2 * nVocab; i++) {
        logits.setAtIndex(ValueLayout.JAVA_FLOAT, i, i);
      }

      var candidates = new LlamaTokenDataArray(arena, nVocab);
      // Filling from the second row must reset size to nVocab and not overrun.
      candidates.fill(logits, nVocab);
      assertThat(candidates.size()).isEqualTo(nVocab);
      assertThat(candidates.selectedIndex()).isEqualTo(-1L);
    }
  }
}
