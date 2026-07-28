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

import static io.gravitee.llama.cpp.LlamaIterator.eogRampBias;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

/**
 * The budget-aware EOG ramp curve.
 *
 * <p>The shape is the whole point: nothing at all until the soft threshold — so a run with the
 * ramp configured is bit-identical to one without it for most of the budget — then a quadratic
 * climb that stays gentle for a while and only becomes overwhelming at the cap. A linear ramp
 * would start nudging the model into early endings as soon as it crossed the threshold.
 *
 * <p>No model or native libraries required.
 *
 * @author GraviteeSource Team
 */
class EogRampBiasTest {

  private static final float START = 0.75f;
  private static final float MAX_BIAS = 24f;

  private static float bias(int used) {
    return eogRampBias(used, 200, START, MAX_BIAS);
  }

  @Test
  void nothing_happens_before_the_soft_threshold() {
    assertThat(bias(0)).isZero();
    assertThat(bias(100)).isZero();
    assertThat(bias(150)).as("exactly at the threshold").isZero();
  }

  @Test
  void the_cap_gets_the_full_bias() {
    assertThat(bias(200)).isEqualTo(MAX_BIAS);
    assertThat(bias(250)).as("clamped past the cap").isEqualTo(MAX_BIAS);
  }

  @Test
  void the_climb_is_quadratic_not_linear() {
    // Halfway through the ramp a linear curve would give 12; quadratic gives a quarter of the
    // max, keeping the model's own phrasing intact for longer.
    assertThat(bias(175)).isEqualTo(MAX_BIAS * 0.25f, within(0.01f));
    assertThat(bias(190)).isEqualTo(MAX_BIAS * 0.64f, within(0.01f));
  }

  @Test
  void the_curve_is_monotonic() {
    float previous = -1f;
    for (int used = 0; used <= 200; used++) {
      float current = bias(used);
      assertThat(current).as("used=%d", used).isGreaterThanOrEqualTo(previous);
      previous = current;
    }
  }

  @Test
  void a_negative_start_fraction_disables_the_ramp() {
    assertThat(eogRampBias(199, 200, -1f, MAX_BIAS)).isZero();
  }

  @Test
  void an_unset_budget_disables_the_ramp() {
    assertThat(eogRampBias(500, -1, START, MAX_BIAS)).isZero();
    assertThat(eogRampBias(500, 0, START, MAX_BIAS)).isZero();
  }

  @Test
  void a_start_fraction_of_one_biases_only_at_the_cap() {
    assertThat(eogRampBias(199, 200, 1f, MAX_BIAS)).isZero();
    assertThat(eogRampBias(200, 200, 1f, MAX_BIAS)).isEqualTo(MAX_BIAS);
  }

  @Test
  void a_start_fraction_of_zero_ramps_across_the_whole_budget() {
    assertThat(eogRampBias(0, 200, 0f, MAX_BIAS)).isZero();
    assertThat(eogRampBias(100, 200, 0f, MAX_BIAS)).isEqualTo(
      MAX_BIAS * 0.25f,
      within(0.01f)
    );
    assertThat(eogRampBias(200, 200, 0f, MAX_BIAS)).isEqualTo(MAX_BIAS);
  }
}
