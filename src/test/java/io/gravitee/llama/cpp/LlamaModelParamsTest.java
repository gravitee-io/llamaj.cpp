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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_supports_gpu_offload;
import static io.gravitee.llama.cpp.SplitMode.LAYER;
import static io.gravitee.llama.cpp.SplitMode.NONE;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.util.Arrays;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class LlamaModelParamsTest extends LlamaCppTest {

  @BeforeAll
  public static void init() {
    LlamaLibLoader.load();
  }

  @Test
  void should_create_LlamaModelParams_with_custom() {
    try (Arena arena = Arena.ofConfined()) {
      long maxDevices = LlamaRuntime.llama_max_devices();
      int mainGpu = (int) (maxDevices / 2);
      var gpuEnabled = llama_supports_gpu_offload();

      var modelParams = new LlamaModelParams(arena)
        .nGpuLayers(gpuEnabled ? 99 : 0)
        .splitMode(LAYER)
        .mainGpu(mainGpu)
        .vocabOnly(true)
        .useMmap(false)
        .useMlock(true)
        .checkTensors(true);

      float[] tensorSplit = modelParams.buildDefaultTensorSplit();
      modelParams.tensorSplit(arena, tensorSplit);

      assertThat(modelParams.nGpuLayers()).isEqualTo(gpuEnabled ? 99 : 0);
      assertThat(modelParams.splitMode()).isEqualTo(LAYER);
      assertThat(modelParams.mainGpu()).isEqualTo(mainGpu);
      assertThat(modelParams.tensorSplit()).containsExactly(tensorSplit);
      assertThat(modelParams.vocabOnly()).isTrue();
      assertThat(modelParams.useMmap()).isFalse();
      assertThat(modelParams.useMlock()).isTrue();
      assertThat(modelParams.checkTensors()).isTrue();
    }
  }

  @Test
  void should_treat_native_load_mode_auto_as_mmap() {
    // llama.cpp v0.4.0 defaults load_mode to LLAMA_LOAD_MODE_AUTO (-1), which the
    // loader resolves to mmap. Toggling bits must never produce an invalid enum.
    try (Arena arena = Arena.ofConfined()) {
      var params = new LlamaModelParams(arena);
      assertThat(params.useMmap()).isTrue();
      assertThat(params.useMlock()).isFalse();

      params.useMlock(true);
      assertThat(params.useMmap()).isTrue();
      assertThat(params.useMlock()).isTrue();

      params.useMlock(false).useMmap(false);
      assertThat(params.useMmap()).isFalse();
      assertThat(params.useMlock()).isFalse();
    }
  }

  @Test
  void should_expose_lazy_mode() {
    try (Arena arena = Arena.ofConfined()) {
      var params = new LlamaModelParams(arena);
      assertThat(params.lazyMode()).isEqualTo(LlamaLazyMode.AUTO);
      params.lazyMode(LlamaLazyMode.OFF);
      assertThat(params.lazyMode()).isEqualTo(LlamaLazyMode.OFF);
    }
  }
}
