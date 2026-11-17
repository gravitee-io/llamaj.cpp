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
 * Locks the derived Itanium mangling of the staging (clef) symbol against the name exported by
 * libllama, and checks the reports name every MTP/EAGLE3 C symbol. Pure string logic — no native lib needed —
 * so a typo in a signature declaration fails fast in any environment. Whether the symbols
 * actually resolve against the bundled build is asserted separately by
 * {@code MtpEagle3SpeculativeTest#staging_symbols_resolve_against_bundled_llama_cpp}.
 *
 * @author GraviteeSource Team
 */
class LlamaExtSymbolsTest {

  @Test
  void derived_symbols_match_libllama_exports() {
    // Joint decision head (clef) — verified against a clang++ build of the llama-ext.h declaration.
    assertThat(LlamaExt.BATCH_EXT_SET_DECISION_ORDER.symbol()).isEqualTo(
      "_Z34llama_batch_ext_set_decision_orderP15llama_batch_exti20llama_decision_order"
    );
  }

  @Test
  void resolution_report_names_every_declared_symbol() {
    // Every declared symbol appears in its report, resolved or not.
    String mtp = LlamaExt.resolutionReport();
    assertThat(mtp)
      .contains("llama_set_embeddings_nextn")
      .contains("llama_get_embeddings_nextn_ith")
      .contains("llama_get_ctx_other");

    String eagle3 = LlamaExt.eagle3ResolutionReport();
    assertThat(eagle3)
      .contains("llama_set_embeddings_layer_inp")
      .contains("llama_get_embeddings_layer_inp")
      .contains("llama_get_embeddings_nextn")
      .contains("llama_model_target_layer_ids");

    assertThat(LlamaExt.decisionResolutionReport()).contains(
      "llama_batch_ext_set_decision_order"
    );
  }
}
