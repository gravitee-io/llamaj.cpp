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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_h;

import java.lang.foreign.Arena;
import java.lang.foreign.SegmentAllocator;

/**
 * Java representation of `mtmd_context_params`.
 * This class holds the configuration for a multimodal context.
 * It directly manages a native `mtmd_context_params` MemorySegment.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class MtmdContextParams extends MemorySegmentAware {

  public MtmdContextParams(Arena arena) {
    super(
      llama_h(
        "mtmd_context_params_default",
        new Class[] { SegmentAllocator.class },
        arena
      )
    );
    if (segment == null || segment.address() == 0L) {
      throw new RuntimeException("Failed to allocate mtmd_context_params");
    }
  }

  public boolean useGpu() {
    return LlamaRuntime.mtmd_context_params_use_gpu(segment);
  }

  public MtmdContextParams useGpu(boolean useGpu) {
    LlamaRuntime.mtmd_context_params_use_gpu(segment, useGpu);
    return this;
  }

  public boolean printTimings() {
    return LlamaRuntime.mtmd_context_params_print_timings(segment);
  }

  public MtmdContextParams printTimings(boolean printTimings) {
    LlamaRuntime.mtmd_context_params_print_timings(segment, printTimings);
    return this;
  }

  public int nThreads() {
    return LlamaRuntime.mtmd_context_params_n_threads(segment);
  }

  public MtmdContextParams nThreads(int nThreads) {
    LlamaRuntime.mtmd_context_params_n_threads(segment, nThreads);
    return this;
  }

  public String mediaMarker() {
    return LlamaRuntime.mtmd_context_params_media_marker(segment).getString(0);
  }

  public MtmdContextParams mediaMarker(String mediaMarker) {
    // Media marker is a const char*, so it needs to be allocated in a persistent arena
    // or passed directly. For simplicity, we'll reallocate it here.
    LlamaRuntime.mtmd_context_params_media_marker(
      segment,
      Arena.ofAuto().allocateFrom(mediaMarker)
    );
    return this;
  }

  public FlashAttentionType flashAttnType() {
    return FlashAttentionType.values()[LlamaRuntime.mtmd_context_params_flash_attn_type(
      segment
    )];
  }

  public MtmdContextParams flashAttnType(FlashAttentionType flashAttnType) {
    LlamaRuntime.mtmd_context_params_flash_attn_type(
      segment,
      flashAttnType.ordinal()
    );
    return this;
  }

  public int imageMinTokens() {
    return LlamaRuntime.mtmd_context_params_image_min_tokens(segment);
  }

  public MtmdContextParams imageMinTokens(int imageMinTokens) {
    LlamaRuntime.mtmd_context_params_image_min_tokens(segment, imageMinTokens);
    return this;
  }

  public int imageMaxTokens() {
    return LlamaRuntime.mtmd_context_params_image_max_tokens(segment);
  }

  public MtmdContextParams imageMaxTokens(int imageMaxTokens) {
    LlamaRuntime.mtmd_context_params_image_max_tokens(segment, imageMaxTokens);
    return this;
  }
}
