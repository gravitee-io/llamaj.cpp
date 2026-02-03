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

import java.lang.foreign.Arena;
import java.nio.file.Path;

/**
 * GGUF model dimensions extracted via a metadata-only load ({@code no_alloc=true}).
 *
 * <p>The {@link #loadFrom(Path)} factory opens the GGUF file, reads header/tensor
 * shapes without allocating weight bytes (O(ms)), and frees the model immediately.
 *
 * <p>This record uses only primitive types and llama.cpp FFM bindings —
 * no external dependencies.
 *
 * @param totalWeightBytes exact total weight size in bytes (from {@code llama_model_size})
 * @param nLayers          number of transformer layers
 * @param nHead            number of attention heads
 * @param nHeadKv          number of key-value attention heads
 * @param headDim          per-head hidden dimension ({@code nEmbd / nHead})
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record LlamaModelDims(
  long totalWeightBytes,
  int nLayers,
  int nHead,
  int nHeadKv,
  int headDim
) {
  /**
   * Loads GGUF metadata via {@code no_alloc=true} and returns the model dimensions.
   * The model is freed immediately after reading the header.
   *
   * <p>Uses {@code use_extra_bufts=false} to prevent the CPU_REPACK buffer type
   * from triggering the mmap allocation path, which asserts {@code !no_alloc}.
   * Without this, tensors that cannot be repacked (e.g. q8_0) fall back to the
   * default CPU buffer type whose mmap path requires real allocation — crashing
   * with {@code GGML_ASSERT(!ml.no_alloc) failed}.
   *
   * <p>Also uses {@code use_mmap=false} because the default CPU buffer type with
   * mmap enabled enters the {@code buffer_from_host_ptr} code path, which also
   * asserts {@code !no_alloc}. Disabling mmap forces the loader into the
   * {@code else} branch that correctly creates a dummy zero-size buffer for
   * {@code no_alloc} loads.
   *
   * @param modelPath absolute path to the GGUF file
   * @return model dimensions, never {@code null}
   * @throws RuntimeException if the file cannot be loaded
   */
  public static LlamaModelDims loadFrom(Path modelPath) {
    try (Arena arena = Arena.ofConfined()) {
      var params = new LlamaModelParams(arena)
        .noAlloc(true)
        .useExtraBufferTypes(false)
        .useMmap(false)
        .nGpuLayers(0);
      var model = new LlamaModel(arena, modelPath, params);
      try {
        long totalWeightBytes = LlamaRuntime.llama_model_size(model.segment);
        int nLayers = LlamaRuntime.llama_model_n_layer(model.segment);
        int nHead = LlamaRuntime.llama_model_n_head(model.segment);
        int nHeadKv = LlamaRuntime.llama_model_n_head_kv(model.segment);
        int nEmbd = LlamaRuntime.llama_model_n_embd(model.segment);
        int headDim = (nHead > 0) ? nEmbd / nHead : 64;
        return new LlamaModelDims(
          totalWeightBytes,
          nLayers,
          nHead,
          nHeadKv,
          headDim
        );
      } finally {
        model.free();
      }
    }
  }
}
