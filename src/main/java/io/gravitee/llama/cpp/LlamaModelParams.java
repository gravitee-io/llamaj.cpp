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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_max_devices;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_model_default_params;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_model_params;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.util.Arrays;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaModelParams extends MemorySegmentAware {

  private final int maxDevices;
  private int nThreads;

  public LlamaModelParams(Arena arena) {
    super(llama_model_default_params(arena));
    this.maxDevices = (int) llama_max_devices();
    this.nThreads = Runtime.getRuntime().availableProcessors(); // Default
  }

  public int getNThreads() {
    return nThreads;
  }

  public LlamaModelParams setNThreads(int nThreads) {
    this.nThreads = nThreads;
    return this;
  }

  public float[] buildDefaultTensorSplit() {
    if (maxDevices > 0) {
      float[] tensorSplit = new float[maxDevices];
      Arrays.fill(tensorSplit, 1f / maxDevices);
      return tensorSplit;
    }
    return new float[0];
  }

  public int nGpuLayers() {
    return LlamaRuntime.n_gpu_layers(segment);
  }

  public LlamaModelParams nGpuLayers(int layers) {
    LlamaRuntime.n_gpu_layers(segment, layers);
    return this;
  }

  public SplitMode splitMode() {
    return SplitMode.fromOrdinal(LlamaRuntime.split_mode(segment));
  }

  public LlamaModelParams splitMode(SplitMode mode) {
    LlamaRuntime.split_mode(segment, mode.ordinal());
    return this;
  }

  public int mainGpu() {
    return LlamaRuntime.main_gpu(segment);
  }

  public LlamaModelParams mainGpu(int mainGpu) {
    LlamaRuntime.main_gpu(segment, mainGpu);
    return this;
  }

  public float[] tensorSplit() {
    var memorySegment = LlamaRuntime.tensor_split(segment);
    float[] tensorSplit = new float[maxDevices];
    for (int i = 0; i < maxDevices; i++) {
      tensorSplit[i] = memorySegment.getAtIndex(JAVA_FLOAT, i);
    }
    return tensorSplit;
  }

  public LlamaModelParams tensorSplit(
    SegmentAllocator allocator,
    float[] tensorSplit
  ) {
    var tensorSplitSegment = allocator.allocate(JAVA_FLOAT, maxDevices);
    MemorySegment.copy(
      MemorySegment.ofArray(tensorSplit),
      0,
      tensorSplitSegment,
      0,
      tensorSplit.length * JAVA_FLOAT.byteSize()
    );

    LlamaRuntime.tensor_split(segment, tensorSplitSegment);
    return this;
  }

  public boolean vocabOnly() {
    return LlamaRuntime.vocab_only(segment);
  }

  public LlamaModelParams vocabOnly(boolean vocabOnly) {
    LlamaRuntime.vocab_only(segment, vocabOnly);
    return this;
  }

  public boolean useMmap() {
    return LlamaRuntime.use_mmap(segment);
  }

  public LlamaModelParams useMmap(boolean useMmap) {
    LlamaRuntime.use_mmap(segment, useMmap);
    return this;
  }

  public boolean useMlock() {
    return LlamaRuntime.use_mlock(segment);
  }

  public LlamaModelParams useMlock(boolean useMlock) {
    LlamaRuntime.use_mlock(segment, useMlock);
    return this;
  }

  public boolean checkTensors() {
    return LlamaRuntime.check_tensors(segment);
  }

  public LlamaModelParams checkTensors(boolean checkTensors) {
    LlamaRuntime.check_tensors(segment, checkTensors);
    return this;
  }
}
