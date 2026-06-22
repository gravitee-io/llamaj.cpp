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

import org.junit.jupiter.api.Test;

/**
 * Pure-logic tests for the diffusion algorithm — no model, no native library, not gated.
 * Exercises the schedule math, block partitioning, validation, and early-stop detection
 * that {@link DiffusionGenerator} / {@link BatchDiffusionIterator} rely on, so regressions
 * are caught in normal CI without the multi-GB diffusion GGUF.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class DiffusionCanvasStateTest {

  private static final int MASK = 0;
  private static final int EOG = 99;

  private static DiffusionParams timestep(int maxLength, int steps) {
    return new DiffusionParams().maxLength(maxLength).steps(steps);
  }

  private static DiffusionParams block(int maxLength, int steps, int blockLength) {
    return new DiffusionParams()
      .maxLength(maxLength)
      .steps(steps)
      .schedule(DiffusionTransferSchedule.BLOCK_BASED)
      .blockLength(blockLength);
  }

  private static DiffusionCanvasState canvas(int[] prompt, DiffusionParams params) {
    return new DiffusionCanvasState(0, prompt, MASK, false, params);
  }

  // ── Schedule / block partitioning ───────────────────────────────────

  @Test
  void timestep_schedule_is_a_single_block() {
    var c = canvas(new int[] { 1, 2, 3 }, timestep(16, 8));
    assertThat(c.numBlocks()).isEqualTo(1);
    assertThat(c.stepsPerBlock()).isEqualTo(8);
    // Whole canvas decoded every step.
    assertThat(c.decodeLength(0)).isEqualTo(16);
  }

  @Test
  void block_schedule_partitions_canvas_and_grows_decode_prefix() {
    // maxLength 16, blockLength 4 => 4 blocks; steps 8 => 2 steps/block.
    var c = canvas(new int[] { 1, 2, 3, 4 }, block(16, 8, 4));
    assertThat(c.numBlocks()).isEqualTo(4);
    assertThat(c.stepsPerBlock()).isEqualTo(2);
    // decodeLength = blockEnd = min(nInput + (b+1)*blockLength, maxLength).
    assertThat(c.decodeLength(0)).isEqualTo(8);
    assertThat(c.decodeLength(1)).isEqualTo(12);
    assertThat(c.decodeLength(2)).isEqualTo(16);
    assertThat(c.decodeLength(3)).isEqualTo(16);
  }

  // ── Construction validation ─────────────────────────────────────────

  @Test
  void rejects_maxLength_not_exceeding_prompt() {
    assertThatThrownBy(() -> canvas(new int[] { 1, 2, 3, 4 }, timestep(4, 8)))
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("maxLength");
  }

  @Test
  void rejects_block_length_not_dividing_canvas() {
    assertThatThrownBy(() -> canvas(new int[] { 1, 2 }, block(16, 8, 5)))
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("blockLength");
  }

  @Test
  void rejects_steps_not_multiple_of_block_count() {
    // 16/4 = 4 blocks, but 6 steps is not a multiple of 4.
    assertThatThrownBy(() -> canvas(new int[] { 1, 2 }, block(16, 6, 4)))
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("multiple");
  }

  // ── Transfer-count math ─────────────────────────────────────────────

  @Test
  void transferCounts_distribute_masks_evenly_with_remainder_up_front() {
    int[] counts = DiffusionCanvasState.transferCounts(10, 4);
    assertThat(counts).containsExactly(3, 3, 2, 2);
    assertThat(java.util.Arrays.stream(counts).sum()).isEqualTo(10);
  }

  @Test
  void timestep_transfer_count_commits_everything_on_the_last_step() {
    var params = timestep(16, 8);
    int remaining = 12;
    // Last step (step == totalSteps - 1) transfers all remaining masked positions.
    assertThat(
      DiffusionCanvasState.transferCount(7, 8, remaining, params, null)
    ).isEqualTo(remaining);
    // Earlier steps commit a strict subset.
    assertThat(
      DiffusionCanvasState.transferCount(0, 8, remaining, params, null)
    ).isBetween(0, remaining - 1);
  }

  // ── Early-stop detection (answerComplete) ───────────────────────────

  @Test
  void answerComplete_false_while_generated_region_is_masked() {
    var c = canvas(new int[] { 1, 2, 3, 4 }, timestep(16, 8));
    assertThat(c.answerComplete(t -> t == EOG)).isFalse();
  }

  @Test
  void answerComplete_true_when_committed_prefix_ends_in_eos() {
    var c = canvas(new int[] { 1, 2, 3, 4 }, timestep(16, 8));
    int[] tok = c.tokens();
    // prompt = [0..3]; fill generated prefix with content then an EOS, leave the tail masked.
    tok[4] = 5;
    tok[5] = 6;
    tok[6] = EOG;
    // tok[7..15] remain MASK — irrelevant, answer terminates at the EOS.
    assertThat(c.answerComplete(t -> t == EOG)).isTrue();
  }

  @Test
  void answerComplete_false_when_a_hole_precedes_the_eos() {
    var c = canvas(new int[] { 1, 2, 3, 4 }, timestep(16, 8));
    int[] tok = c.tokens();
    tok[4] = 5;
    // tok[5] stays MASK (a hole)
    tok[6] = EOG;
    assertThat(c.answerComplete(t -> t == EOG)).isFalse();
  }

  // ── done() ──────────────────────────────────────────────────────────

  @Test
  void done_only_when_no_mask_remains() {
    var c = canvas(new int[] { 1, 2 }, timestep(4, 4));
    assertThat(c.done()).isFalse();
    int[] tok = c.tokens();
    tok[2] = 7;
    tok[3] = 8;
    assertThat(c.done()).isTrue();
  }
}
