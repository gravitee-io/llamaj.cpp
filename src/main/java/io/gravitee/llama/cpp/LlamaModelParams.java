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
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.List;

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

  /**
   * When set to {@code true}, the model is loaded in metadata-only mode: GGUF headers
   * and tensor shapes are read but no weight data is allocated or mapped.
   *
   * <p>This is used by {@code LlamaMemoryEstimator} to inspect layer counts and weight
   * sizes from GGUF metadata in O(ms) with zero GPU/RAM footprint. Always call
   * {@link io.gravitee.llama.cpp.LlamaModel#free()} immediately after reading the
   * needed metadata.
   *
   * <p><strong>Important:</strong> when using {@code noAlloc=true}, you should also
   * call {@link #useExtraBufferTypes(boolean)} with {@code false} to prevent the
   * CPU_REPACK buffer type from triggering a tensor allocation assertion in the
   * mmap code path. See {@link LlamaModelDims#loadFrom(java.nio.file.Path)} for
   * the canonical usage pattern.
   *
   * @param noAlloc {@code true} to skip weight allocation (metadata read only).
   * @return this instance for chaining.
   */
  public LlamaModelParams noAlloc(boolean noAlloc) {
    LlamaRuntime.llama_model_params(
      "no_alloc",
      new Class<?>[] { MemorySegment.class, boolean.class },
      segment,
      noAlloc
    );
    return this;
  }

  /**
   * Controls whether extra buffer types (e.g. CPU_REPACK for runtime weight
   * repacking of Q4_0 → Q4_0_4_4) are used during model loading.
   *
   * <p>Defaults to {@code true} in llama.cpp, which enables SIMD-optimized
   * weight layouts on supported CPUs. However, this <b>must</b> be set to
   * {@code false} when {@link #noAlloc(boolean)} is {@code true}, because
   * the CPU_REPACK buffer type causes some tensors to fall back to the
   * default CPU buffer type, which enters the mmap allocation path and
   * asserts {@code !no_alloc} — crashing with
   * {@code GGML_ASSERT(!ml.no_alloc) failed}.
   *
   * @param useExtraBufferTypes {@code false} to disable extra buffer types.
   * @return this instance for chaining.
   */
  public LlamaModelParams useExtraBufferTypes(boolean useExtraBufferTypes) {
    LlamaRuntime.llama_model_params(
      "use_extra_bufts",
      new Class<?>[] { MemorySegment.class, boolean.class },
      segment,
      useExtraBufferTypes
    );
    return this;
  }

  /**
   * Registers remote RPC servers as backend devices for distributed inference.
   * <p>
   * Once registered, llama.cpp will automatically distribute model weights and KV-cache
   * across all available devices (local and remote) in proportion to available memory,
   * unless overridden via {@link #tensorSplit(SegmentAllocator, float[])}.
   * <p>
   * Must be called after {@link LlamaRuntime#ggml_backend_load_all_from_path(Arena, String)}
   * and before loading the model with {@link LlamaModel}.
   *
   * @param arena     The Arena for memory allocation.
   * @param endpoints The RPC server endpoints in "host:port" format (e.g., "192.168.1.10:50052").
   * @return this instance for chaining.
   * @throws IllegalStateException if RPC is not supported by the loaded library.
   */
  public LlamaModelParams rpcServers(Arena arena, String... endpoints) {
    BackendRegistry.addRpcServers(arena, endpoints);
    return this;
  }

  /**
   * Restricts model offloading to only the specified devices.
   * <p>
   * When set, only these devices will be used for GPU layer offloading.
   * This is useful when using RPC to prevent weights from being loaded
   * on local GPU devices (e.g., Metal).
   *
   * @param arena   The Arena for memory allocation.
   * @param devices The device handles to use (from {@link BackendRegistry#listDevices()}).
   * @return this instance for chaining.
   */
  public LlamaModelParams devices(Arena arena, List<MemorySegment> devices) {
    // Allocate a NULL-terminated array of pointers
    var devArray = arena.allocate(ValueLayout.ADDRESS, devices.size() + 1);
    for (int i = 0; i < devices.size(); i++) {
      devArray.setAtIndex(ValueLayout.ADDRESS, i, devices.get(i));
    }
    devArray.setAtIndex(
      ValueLayout.ADDRESS,
      devices.size(),
      MemorySegment.NULL
    );
    LlamaRuntime.devices(segment, devArray);
    return this;
  }
}
