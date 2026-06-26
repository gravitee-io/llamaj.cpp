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
 * Pure mapping tests for {@link GgmlType} — no native library. Guards the
 * non-contiguous native {@code ggml_type} values (which must NOT be derived from
 * {@code ordinal()}).
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class GgmlTypeTest {

  @Test
  void native_values_match_ggml_type_and_are_non_contiguous() {
    assertThat(GgmlType.F32.nativeValue()).isEqualTo(0);
    assertThat(GgmlType.F16.nativeValue()).isEqualTo(1);
    assertThat(GgmlType.Q4_0.nativeValue()).isEqualTo(2);
    assertThat(GgmlType.Q4_1.nativeValue()).isEqualTo(3);
    // Gap: Q4_2/Q4_3 were removed upstream, so Q5_0 jumps to 6.
    assertThat(GgmlType.Q5_0.nativeValue()).isEqualTo(6);
    assertThat(GgmlType.Q5_1.nativeValue()).isEqualTo(7);
    assertThat(GgmlType.Q8_0.nativeValue()).isEqualTo(8);
    assertThat(GgmlType.IQ4_NL.nativeValue()).isEqualTo(20);
    assertThat(GgmlType.BF16.nativeValue()).isEqualTo(30);
  }

  @Test
  void native_value_differs_from_ordinal_for_gapped_types() {
    // If this ever coincides it's a coincidence — the point is we must not use ordinal().
    assertThat(GgmlType.Q5_0.nativeValue()).isNotEqualTo(
      GgmlType.Q5_0.ordinal()
    );
    assertThat(GgmlType.BF16.nativeValue()).isNotEqualTo(
      GgmlType.BF16.ordinal()
    );
  }

  @Test
  void fromNative_round_trips() {
    for (GgmlType type : GgmlType.values()) {
      assertThat(GgmlType.fromNative(type.nativeValue())).isEqualTo(type);
    }
  }

  @Test
  void fromNative_rejects_unknown_value() {
    assertThatThrownBy(() -> GgmlType.fromNative(999)).isInstanceOf(
      LlamaException.class
    );
  }

  @Test
  void isQuantized_is_true_only_for_quant_types() {
    assertThat(GgmlType.F32.isQuantized()).isFalse();
    assertThat(GgmlType.F16.isQuantized()).isFalse();
    assertThat(GgmlType.BF16.isQuantized()).isFalse();
    assertThat(GgmlType.Q8_0.isQuantized()).isTrue();
    assertThat(GgmlType.Q4_0.isQuantized()).isTrue();
    assertThat(GgmlType.IQ4_NL.isQuantized()).isTrue();
  }
}
