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

import static io.gravitee.llama.cpp.LlamaRuntime.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaContext extends MemorySegmentAware implements Freeable {

  private final int nCtx;
  private final int nBatch;
  private final int nSeqMax;
  private final LlamaMemory memory;

  public LlamaContext(LlamaModel model, LlamaContextParams params) {
    super(llama_init_from_model(model.segment, params.segment));
    if (segment == null || segment.address() == 0) {
      throw new LlamaException("Failed to create context from model");
    }
    this.nCtx = params.nCtx();
    this.nBatch = params.nBatch();
    this.nSeqMax = params.nSeqMax();
    this.memory = new LlamaMemory(this);
  }

  public int nCtx() {
    return nCtx;
  }

  public int nBatch() {
    return nBatch;
  }

  public int nSeqMax() {
    return nSeqMax;
  }

  public int nCtxUsedCells() {
    return memory.posMax() - memory.posMin() + 1;
  }

  public void clearCache() {
    checkNotFreed();
    memory.clear();
  }

  public LlamaMemory getMemory() {
    checkNotFreed();
    return memory;
  }

  public LlamaPerformance.ContextPerformance getPerformance(Arena arena) {
    checkNotFreed();
    MemorySegment perfData = llama_perf_context(arena, segment);
    return new LlamaPerformance.ContextPerformance(
      llama_perf_context_t_start_ms(perfData),
      llama_perf_context_t_load_ms(perfData),
      llama_perf_context_t_p_eval_ms(perfData),
      llama_perf_context_t_eval_ms(perfData),
      llama_perf_context_n_p_eval(perfData),
      llama_perf_context_n_eval(perfData),
      llama_perf_context_n_reused(perfData)
    );
  }

  @Override
  public void free() {
    checkNotFreed();
    markFreed();
    llama_free(this);
  }
}
