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

import static io.gravitee.llama.cpp.LlamaRuntime.ggml_backend_reg_count;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.assertThatThrownBy;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Tests for multimodal context and image processing.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class MtmdContextTest extends LlamaCppTest {

  private static Arena arena;
  private static LlamaModel model;
  private static LlamaContext llamaContext;

  @BeforeAll
  public static void beforeAll() throws IOException {
    arena = Arena.ofAuto();

    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

    System.out.println("****************************");
    System.out.println("Libraries loaded at: " + libPath);
    System.out.println(
      "Number of devices registered: " + ggml_backend_reg_count()
    );
    System.out.println("****************************");
  }

  @Test
  void should_create_mtmd_context_params_with_custom_values() {
    var params = new MtmdContextParams(arena)
      .useGpu(true)
      .printTimings(true)
      .nThreads(4)
      .mediaMarker("<IMG>")
      .flashAttnType(FlashAttentionType.ENABLED)
      .imageMinTokens(10)
      .imageMaxTokens(100);

    assertThat(params.useGpu()).isTrue();
    assertThat(params.printTimings()).isTrue();
    assertThat(params.nThreads()).isEqualTo(4);
    assertThat(params.mediaMarker()).isEqualTo("<IMG>");
    assertThat(params.flashAttnType()).isEqualTo(FlashAttentionType.ENABLED);
    assertThat(params.imageMinTokens()).isEqualTo(10);
    assertThat(params.imageMaxTokens()).isEqualTo(100);
  }
}
