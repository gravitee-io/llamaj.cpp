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

import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_TO_DOWNLOAD;
import static io.gravitee.llama.cpp.LlamaCppTest.getModelPath;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

/**
 * Tests for the metadata introspection and dimension APIs added to {@link LlamaModel}.
 * Uses the standard generative GGUF already downloaded by the test suite — no embedding
 * model is required.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class LlamaModelTest {

  private static Arena arena;
  private static LlamaModel model;

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);
    var modelPath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    // Load without vocabOnly so that embedding dimensions and model description
    // are fully populated by llama.cpp (vocabOnly skips tensor loading and zeros
    // out hparams such as n_embd, making nEmbdOut() and desc() return empty values).
    var modelParams = new LlamaModelParams(arena);
    model = new LlamaModel(arena, modelPath, modelParams);
  }

  @AfterAll
  static void afterAll() {
    if (model != null) {
      model.free();
    }
    LlamaRuntime.llama_backend_free();
    arena.close();
  }

  @Test
  @Order(1)
  void nEmbdOut_should_be_positive() {
    assertThat(model.nEmbdOut()).isGreaterThan(0);
  }

  @Test
  @Order(2)
  void nClsOut_should_be_one_for_generative_model() {
    // llama.cpp defaults n_cls_out to 1 for all non-classifier models.
    // Only dedicated classifier / reranker GGUFs expose values > 1.
    assertThat(model.nClsOut()).isEqualTo(1);
  }

  @Test
  @Order(3)
  void metaCount_should_be_positive() {
    assertThat(model.metaCount()).isGreaterThan(0);
  }

  @Test
  @Order(4)
  void metaVal_should_return_architecture_for_standard_key() {
    try (Arena localArena = Arena.ofConfined()) {
      String arch = model.metaVal(localArena, "general.architecture");
      assertThat(arch).isNotNull().isNotBlank();
    }
  }

  @Test
  @Order(5)
  void metaVal_should_return_null_for_unknown_key() {
    try (Arena localArena = Arena.ofConfined()) {
      String val = model.metaVal(localArena, "this.key.does.not.exist");
      assertThat(val).isNull();
    }
  }

  @Test
  @Order(6)
  void meta_should_return_non_empty_map_containing_architecture() {
    try (Arena localArena = Arena.ofConfined()) {
      Map<String, String> meta = model.meta(localArena);
      assertThat(meta).isNotEmpty();
      assertThat(meta).containsKey("general.architecture");
      assertThat(meta.get("general.architecture")).isNotBlank();
    }
  }

  @Test
  @Order(7)
  void meta_size_should_match_metaCount() {
    try (Arena localArena = Arena.ofConfined()) {
      int count = model.metaCount();
      Map<String, String> meta = model.meta(localArena);
      assertThat(meta).hasSize(count);
    }
  }

  @Test
  @Order(8)
  void desc_should_return_non_blank_string() {
    try (Arena localArena = Arena.ofConfined()) {
      String desc = model.desc(localArena);
      assertThat(desc).isNotNull().isNotBlank();
    }
  }
}
