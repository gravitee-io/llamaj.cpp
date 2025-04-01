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

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;

import static io.gravitee.llama.cpp.LlamaRuntime.llama_supports_gpu_offload;
import static io.gravitee.llama.cpp.SplitMode.LAYER;
import static io.gravitee.llama.cpp.SplitMode.NONE;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class LlamaModelParamsTest extends LlamaCppTest {

    @Test
    void should_create_LlamaModelParams_with_default() {
        try (Arena arena = Arena.ofConfined()) {
            var modelParams = new LlamaModelParams(arena);
            assertThat(modelParams.nGpuLayers()).isGreaterThanOrEqualTo(0);
            assertThat(modelParams.splitMode()).isEqualTo(LAYER);
            assertThat(modelParams.mainGpu()).isEqualTo(0);
            assertThat(modelParams.tensorSplit()).containsExactly(getEvenSplit(LlamaRuntime.llama_max_devices()));
            assertThat(modelParams.vocabOnly()).isFalse();
            assertThat(modelParams.useMmap()).isTrue();
            assertThat(modelParams.useMlock()).isFalse();
            assertThat(modelParams.checkTensors()).isFalse();
        }
    }

    private static float[] getEvenSplit(long maxDevices) {
        var tensorSplits = new float[(int) maxDevices];
        for (int i = 0; i < maxDevices; i++) {
            tensorSplits[i] = 1f / maxDevices;
        }
        return tensorSplits;
    }

    @Test
    void should_create_LlamaModelParams_with_custom() {
        try (Arena arena = Arena.ofConfined()) {
            long maxDevices = LlamaRuntime.llama_max_devices();
            int mainGpu = (int) (maxDevices / 2);
            float[] biasSplit = getBiasSplit(maxDevices, mainGpu);
            var gpuEnabled = llama_supports_gpu_offload();

            var modelParams = new LlamaModelParams(arena)
                    .nGpuLayers(gpuEnabled ? 99 : 0)
                    .splitMode(NONE)
                    .mainGpu(mainGpu)
                    .tensorSplit(arena, biasSplit)
                    .vocabOnly(true)
                    .useMmap(false)
                    .useMlock(true)
                    .checkTensors(true);

            assertThat(modelParams.nGpuLayers()).isEqualTo(99);
            assertThat(modelParams.splitMode()).isEqualTo(NONE);
            assertThat(modelParams.mainGpu()).isEqualTo(mainGpu);
            assertThat(modelParams.tensorSplit()).containsExactly(biasSplit);
            assertThat(modelParams.vocabOnly()).isTrue();
            assertThat(modelParams.useMmap()).isFalse();
            assertThat(modelParams.useMlock()).isTrue();
            assertThat(modelParams.checkTensors()).isTrue();
        }
    }

    private static float[] getBiasSplit(long maxDevices, int mainGpu) {
        var tensorSplits = new float[(int) maxDevices];
        tensorSplits[mainGpu] = 1f;
        return tensorSplits;
    }
}
