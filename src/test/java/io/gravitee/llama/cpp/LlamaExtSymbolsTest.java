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
 * Locks {@link LlamaExt.Fn}'s derived Itanium mangling against the symbol names observed in the
 * pinned libllama ({@code nm -gU libllama.dylib}). Pure string logic — no native lib needed —
 * so a typo in a signature declaration fails fast in any environment. Whether the symbols
 * actually resolve against the bundled build is asserted separately by
 * {@code MtpEagle3SpeculativeTest#staging_symbols_resolve_against_bundled_llama_cpp}.
 *
 * @author GraviteeSource Team
 */
class LlamaExtSymbolsTest {

  @Test
  void derived_symbols_match_libllama_exports() {
    // MTP (nextn) group — verified against libllama b9673 and b9873.
    assertThat(LlamaExt.SET_EMBEDDINGS_NEXTN.symbol()).isEqualTo(
      "_Z26llama_set_embeddings_nextnP13llama_contextbb"
    );
    assertThat(LlamaExt.GET_EMBEDDINGS_NEXTN_ITH.symbol()).isEqualTo(
      "_Z30llama_get_embeddings_nextn_ithP13llama_contexti"
    );
    assertThat(LlamaExt.GET_CTX_OTHER.symbol()).isEqualTo(
      "_Z19llama_get_ctx_otherP13llama_context"
    );

    // EAGLE3 group — verified against libllama b9873.
    assertThat(LlamaExt.SET_EMBEDDINGS_LAYER_INP.symbol()).isEqualTo(
      "_Z30llama_set_embeddings_layer_inpP13llama_contextjb"
    );
    assertThat(LlamaExt.GET_EMBEDDINGS_LAYER_INP.symbol()).isEqualTo(
      "_Z30llama_get_embeddings_layer_inpP13llama_contextj"
    );
    assertThat(LlamaExt.GET_EMBEDDINGS_NEXTN.symbol()).isEqualTo(
      "_Z26llama_get_embeddings_nextnP13llama_context"
    );
    assertThat(LlamaExt.MODEL_TARGET_LAYER_IDS.symbol()).isEqualTo(
      "_Z28llama_model_target_layer_idsPK11llama_model"
    );
    assertThat(LlamaExt.MODEL_TARGET_LAYER_IDS_N.symbol()).isEqualTo(
      "_Z30llama_model_target_layer_ids_nPK11llama_model"
    );
  }

  @Test
  void resolution_report_names_every_declared_symbol() {
    // Report generation is pure over the group lists; every declared symbol must appear.
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
  }
}
