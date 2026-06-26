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
 * Pure tests for {@link SpeculativeConfig} — no native library, no model. Cover the adaptive
 * draft-length parameters, backward-compatible construction, and {@code draftMin} clamping.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class SpeculativeConfigTest {

  @Test
  void five_arg_constructor_is_non_adaptive() {
    var config = new SpeculativeConfig(4, 0.8f, 40, 0.95f, 42);
    // Backward-compatible ctor drafts a fixed window: draftMin == nDraft, pMin == 0.
    assertThat(config.nDraft()).isEqualTo(4);
    assertThat(config.draftMin()).isEqualTo(4);
    assertThat(config.pMin()).isZero();
    assertThat(config.isAdaptive()).isFalse();
    assertThat(config.isGreedy()).isFalse();
  }

  @Test
  void greedy_factory_is_greedy_and_non_adaptive() {
    var config = SpeculativeConfig.greedy(4);
    assertThat(config.isGreedy()).isTrue();
    assertThat(config.isAdaptive()).isFalse();
    assertThat(config.draftMin()).isEqualTo(4);
  }

  @Test
  void greedy_adaptive_factory_enables_early_stop() {
    var config = SpeculativeConfig.greedyAdaptive(8, 2, 0.6f);
    assertThat(config.isGreedy()).isTrue();
    assertThat(config.isAdaptive()).isTrue();
    assertThat(config.nDraft()).isEqualTo(8);
    assertThat(config.draftMin()).isEqualTo(2);
    assertThat(config.pMin()).isEqualTo(0.6f);
  }

  @Test
  void draft_min_is_clamped_to_range() {
    // Below 1 clamps up to 1.
    assertThat(
      SpeculativeConfig.greedyAdaptive(8, 0, 0.5f).draftMin()
    ).isEqualTo(1);
    assertThat(
      SpeculativeConfig.greedyAdaptive(8, -5, 0.5f).draftMin()
    ).isEqualTo(1);
    // Above nDraft clamps down to nDraft.
    assertThat(
      SpeculativeConfig.greedyAdaptive(4, 9, 0.5f).draftMin()
    ).isEqualTo(4);
  }

  @Test
  void zero_or_negative_p_min_disables_adaptive() {
    assertThat(
      new SpeculativeConfig(4, 0.0f, 0, 1.0f, 0L, 2, 0.0f).isAdaptive()
    ).isFalse();
    assertThat(
      new SpeculativeConfig(4, 0.0f, 0, 1.0f, 0L, 2, -0.1f).isAdaptive()
    ).isFalse();
  }

  @Test
  void n_draft_below_one_is_rejected() {
    assertThatThrownBy(() -> SpeculativeConfig.greedy(0)).isInstanceOf(
      LlamaException.class
    );
  }

  @Test
  void withers_override_one_field_and_preserve_the_rest() {
    var config = SpeculativeConfig.greedy(8)
      .withTemperature(0.8f)
      .withTopK(40)
      .withTopP(0.95f)
      .withSeed(42L);

    assertThat(config.nDraft()).isEqualTo(8); // preserved from greedy(8)
    assertThat(config.temperature()).isEqualTo(0.8f);
    assertThat(config.topK()).isEqualTo(40);
    assertThat(config.topP()).isEqualTo(0.95f);
    assertThat(config.seed()).isEqualTo(42L);
    assertThat(config.isGreedy()).isFalse(); // temperature made it a sampling config
    assertThat(config.isNgram()).isFalse();
  }

  @Test
  void with_ngram_switches_to_prompt_lookup() {
    var config = SpeculativeConfig.greedy(4).withNgram(2);
    assertThat(config.isNgram()).isTrue();
    assertThat(config.ngram()).isEqualTo(2);
    assertThat(config.isGreedy()).isTrue();
  }

  @Test
  void withers_revalidate_via_canonical_constructor() {
    // withNgram(-1) must go through the canonical constructor's validation.
    assertThatThrownBy(() ->
      SpeculativeConfig.greedy(4).withNgram(-1)
    ).isInstanceOf(LlamaException.class);
  }

  @Test
  void builder_defaults_to_greedy_model_drafting() {
    var config = SpeculativeConfig.builder().build();
    assertThat(config.nDraft()).isEqualTo(SpeculativeConfig.DEFAULT_N_DRAFT);
    assertThat(config.isGreedy()).isTrue();
    assertThat(config.isNgram()).isFalse();
    assertThat(config.isAdaptive()).isFalse();
  }

  @Test
  void builder_sets_each_field() {
    var config = SpeculativeConfig.builder()
      .nDraft(8)
      .temperature(0.8f)
      .topK(40)
      .topP(0.95f)
      .seed(42L)
      .build();
    assertThat(config.nDraft()).isEqualTo(8);
    assertThat(config.temperature()).isEqualTo(0.8f);
    assertThat(config.topK()).isEqualTo(40);
    assertThat(config.topP()).isEqualTo(0.95f);
    assertThat(config.seed()).isEqualTo(42L);
    assertThat(config.isGreedy()).isFalse();
  }

  @Test
  void to_builder_round_trips() {
    var original = SpeculativeConfig.greedyAdaptive(8, 2, 0.6f);
    var copy = original.toBuilder().build();
    assertThat(copy).isEqualTo(original);
  }

  @Test
  void builder_validates_on_build() {
    assertThatThrownBy(() ->
      SpeculativeConfig.builder().nDraft(0).build()
    ).isInstanceOf(LlamaException.class);
  }
}
