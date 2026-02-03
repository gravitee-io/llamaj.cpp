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
import static io.gravitee.llama.cpp.LlamaCppTest.REASONING_MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.REASONNING_MODEL_TO_DOWNLOAD;
import static io.gravitee.llama.cpp.LlamaCppTest.getModelPath;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

/**
 * Tests for {@link LlamaModelDims} metadata-only model loading.
 *
 * <p>These tests verify that {@link LlamaModelDims#loadFrom(Path)} correctly reads
 * GGUF model metadata without allocating tensors, and that it does not crash with
 * {@code GGML_ASSERT(!ml.no_alloc)} even when compute backends (CPU_REPACK, Metal,
 * etc.) are globally registered.
 *
 * <p>The test deliberately registers all backends <b>before</b> calling {@code loadFrom}
 * to reproduce the production scenario where {@code ggml_backend_load_all_from_path}
 * has already run (e.g. from a previous model load in the same JVM).
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class LlamaModelDimsTest {

  private static Arena arena;

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();

    // Reproduce the production scenario: load all backends globally (CPU, Metal, RPC, etc.)
    // This registers CPU_REPACK as an extra buffer type on the CPU backend.
    // Before the fix, calling LlamaModelDims.loadFrom() after this would crash with:
    //   GGML_ASSERT(!ml.no_alloc) failed
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);
  }

  @AfterAll
  static void afterAll() {
    LlamaRuntime.llama_backend_free();
    arena.close();
  }

  @Test
  @Order(1)
  @DisplayName(
    "loadFrom should not crash with GGML_ASSERT when backends are registered"
  )
  void loadFrom_should_not_crash_with_registered_backends() {
    Path modelPath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);

    // This would previously crash with:
    //   load_tensors: tensor 'token_embd.weight' (q8_0) cannot be used with
    //   preferred buffer type CPU_REPACK, using CPU instead
    //   GGML_ASSERT(!ml.no_alloc) failed
    assertThatCode(() ->
      LlamaModelDims.loadFrom(modelPath)
    ).doesNotThrowAnyException();
  }

  @Test
  @Order(2)
  @DisplayName("loadFrom should return valid dimensions for a standard model")
  void loadFrom_should_return_valid_dimensions() {
    Path modelPath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);

    LlamaModelDims dims = LlamaModelDims.loadFrom(modelPath);

    // Llama-3.2-1B-Instruct-IQ3_M: ~1B params, 16 layers, 32 heads, 8 kv heads
    assertThat(dims.totalWeightBytes()).isGreaterThan(0);
    assertThat(dims.nLayers()).isGreaterThan(0);
    assertThat(dims.nHead()).isGreaterThan(0);
    assertThat(dims.nHeadKv()).isGreaterThan(0);
    assertThat(dims.headDim()).isGreaterThan(0);

    System.out.printf(
      "Model dims: weightBytes=%,d, nLayers=%d, nHead=%d, nHeadKv=%d, headDim=%d%n",
      dims.totalWeightBytes(),
      dims.nLayers(),
      dims.nHead(),
      dims.nHeadKv(),
      dims.headDim()
    );
  }

  @Test
  @Order(3)
  @DisplayName(
    "loadFrom should return valid dimensions for a Q8_0 reasoning model"
  )
  void loadFrom_should_return_valid_dimensions_for_q8_model() {
    // Q8_0 models are specifically affected by CPU_REPACK because q8_0 tensors
    // cannot be repacked and fall back to the default CPU buffer type
    Path modelPath = getModelPath(
      REASONING_MODEL_PATH,
      REASONNING_MODEL_TO_DOWNLOAD
    );

    LlamaModelDims dims = LlamaModelDims.loadFrom(modelPath);

    // Qwen3-0.6B-Q8_0: ~0.6B params, 28 layers
    assertThat(dims.totalWeightBytes()).isGreaterThan(0);
    assertThat(dims.nLayers()).isGreaterThan(0);
    assertThat(dims.nHead()).isGreaterThan(0);
    assertThat(dims.nHeadKv()).isGreaterThan(0);
    assertThat(dims.headDim()).isGreaterThan(0);

    System.out.printf(
      "Reasoning model dims: weightBytes=%,d, nLayers=%d, nHead=%d, nHeadKv=%d, headDim=%d%n",
      dims.totalWeightBytes(),
      dims.nLayers(),
      dims.nHead(),
      dims.nHeadKv(),
      dims.headDim()
    );
  }

  @Test
  @Order(4)
  @DisplayName(
    "loadFrom can be called multiple times without crash (idempotent)"
  )
  void loadFrom_should_be_idempotent() {
    Path modelPath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);

    LlamaModelDims first = LlamaModelDims.loadFrom(modelPath);
    LlamaModelDims second = LlamaModelDims.loadFrom(modelPath);

    assertThat(first).isEqualTo(second);
  }

  @Test
  @Order(5)
  @DisplayName(
    "loadFrom followed by full model load should work (no state corruption)"
  )
  void loadFrom_should_not_corrupt_backend_state_for_subsequent_full_load() {
    Path modelPath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);

    // First: metadata-only load (the estimator path)
    LlamaModelDims dims = LlamaModelDims.loadFrom(modelPath);
    assertThat(dims.totalWeightBytes()).isGreaterThan(0);

    // Second: full model load (the real inference path) — should still work
    var modelParams = new LlamaModelParams(arena);
    var model = new LlamaModel(arena, modelPath, modelParams);
    try {
      long modelSize = LlamaRuntime.llama_model_size(model.segment);
      assertThat(modelSize).isEqualTo(dims.totalWeightBytes());
    } finally {
      model.free();
    }
  }

  @Test
  @Order(6)
  @DisplayName(
    "useExtraBufferTypes(false) should be settable on LlamaModelParams"
  )
  void useExtraBufferTypes_should_be_configurable() {
    try (Arena localArena = Arena.ofConfined()) {
      var params = new LlamaModelParams(localArena)
        .noAlloc(true)
        .useExtraBufferTypes(false)
        .useMmap(false)
        .nGpuLayers(0);

      // Verify the params were set (we can't read use_extra_bufts back directly
      // through our API, but we can verify it doesn't crash when used)
      assertThat(params.nGpuLayers()).isEqualTo(0);
      assertThat(params.useMmap()).isFalse();
    }
  }
}
