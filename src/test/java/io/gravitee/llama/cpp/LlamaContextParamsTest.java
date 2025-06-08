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

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

import java.lang.foreign.Arena;
import java.lang.runtime.TemplateRuntime;
import org.junit.jupiter.api.Test;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class LlamaContextParamsTest extends LlamaCppTest {

  @Test
  void should_create_LlamaContextParams_with_default() {
    try (Arena arena = Arena.ofConfined()) {
      var contextParams = new LlamaContextParams(arena);
      assertThat(contextParams.nCtx()).isEqualTo(512);
      assertThat(contextParams.nBatch()).isEqualTo(2048);
      assertThat(contextParams.nUBatch()).isEqualTo(512);
      assertThat(contextParams.nSeqMax()).isEqualTo(1);
      assertThat(contextParams.nThreads()).isEqualTo(4);
      assertThat(contextParams.nThreadsBatch()).isEqualTo(4);
      assertThat(contextParams.poolingType()).isEqualTo(PoolingType.UNSPECIFIED);
      assertThat(contextParams.attentionType()).isEqualTo(AttentionType.UNSPECIFIED);
      assertThat(contextParams.embeddings()).isFalse();
      assertThat(contextParams.offloadKQV()).isTrue();
      assertThat(contextParams.flashAttn()).isFalse();
      assertThat(contextParams.noPerf()).isTrue();
    }
  }

  @Test
  void should_create_LlamaContextParams_with_custom() {
    try (Arena arena = Arena.ofConfined()) {
      var contextParams = new LlamaContextParams(arena)
        .nCtx(99)
        .nBatch(42)
        .nUBatch(64)
        .nSeqMax(6)
        .nThreads(16)
        .nThreadsBatch(14)
        .poolingType(PoolingType.CLS)
        .attentionType(AttentionType.CAUSAL)
        .embeddings(true)
        .offloadKQV(false)
        .flashAttn(true)
        .noPerf(true);

      assertThat(contextParams.nCtx()).isEqualTo(99);
      assertThat(contextParams.nBatch()).isEqualTo(42);
      assertThat(contextParams.nUBatch()).isEqualTo(64);
      assertThat(contextParams.nSeqMax()).isEqualTo(6);
      assertThat(contextParams.nThreads()).isEqualTo(16);
      assertThat(contextParams.nThreadsBatch()).isEqualTo(14);
      assertThat(contextParams.poolingType()).isEqualTo(PoolingType.CLS);
      assertThat(contextParams.attentionType()).isEqualTo(AttentionType.CAUSAL);
      assertThat(contextParams.embeddings()).isTrue();
      assertThat(contextParams.offloadKQV()).isFalse();
      assertThat(contextParams.flashAttn()).isTrue();
      assertThat(contextParams.noPerf()).isTrue();
    }
  }
}
