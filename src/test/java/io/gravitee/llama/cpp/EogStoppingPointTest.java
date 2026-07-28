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

import static io.gravitee.llama.cpp.LlamaIterator.atStoppingPoint;
import static io.gravitee.llama.cpp.LlamaIterator.eogRampProgress;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

/**
 * The boundary gate on the budget-aware EOG ramp.
 *
 * <p>Without it the ramp was measured cutting mid-clause — the same severed output the budget was
 * meant to avoid, only earlier. The gate is what makes the bias a landing: EOG can
 * only win where the text had already reached an exit.
 *
 * <p>No model or native libraries required.
 *
 * @author GraviteeSource Team
 */
class EogStoppingPointTest {

  @Test
  void nothing_is_a_stopping_point_before_any_output() {
    assertThat(atStoppingPoint((char) 0, false, 0.9f)).isFalse();
  }

  @Test
  void sentence_punctuation_stops_from_the_very_start_of_the_ramp() {
    for (char c : ".!?…。".toCharArray()) {
      assertThat(atStoppingPoint(c, false, 0f)).as("'%c'", c).isTrue();
    }
  }

  @Test
  void a_line_break_is_a_stopping_point() {
    // The only boundary structured output offers: table rows and list items rarely end in a
    // full stop, and that is exactly where the ungated ramp used to cut.
    assertThat(atStoppingPoint('|', true, 0f)).isTrue();
  }

  @Test
  void clause_punctuation_waits_for_the_second_half() {
    assertThat(atStoppingPoint(',', false, 0.4f)).isFalse();
    assertThat(atStoppingPoint(',', false, 0.5f)).isTrue();
    assertThat(atStoppingPoint(';', false, 0.8f)).isTrue();
  }

  @Test
  void mid_word_is_never_a_stopping_point_until_the_cap() {
    assertThat(atStoppingPoint('t', false, 0.99f)).isFalse();
    assertThat(atStoppingPoint('t', false, 1f))
      .as("the cap is the backstop — the hard limit would sever it here anyway")
      .isTrue();
  }

  @Test
  void progress_is_zero_below_the_threshold_and_one_at_the_cap() {
    assertThat(eogRampProgress(100, 200, 0.75f)).isZero();
    assertThat(eogRampProgress(175, 200, 0.75f)).isEqualTo(0.5f, within(0.01f));
    assertThat(eogRampProgress(200, 200, 0.75f)).isEqualTo(1f);
    assertThat(eogRampProgress(999, 200, 0.75f)).isEqualTo(1f);
  }

  @Test
  void progress_is_zero_when_the_ramp_is_disabled() {
    assertThat(eogRampProgress(199, 200, -1f)).isZero();
    assertThat(eogRampProgress(199, -1, 0.75f)).isZero();
  }
}
