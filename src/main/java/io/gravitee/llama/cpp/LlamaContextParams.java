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

import static io.gravitee.llama.cpp.LlamaRuntime.attention_type;
import static io.gravitee.llama.cpp.LlamaRuntime.flash_attn;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_context_default_params;
import static io.gravitee.llama.cpp.LlamaRuntime.n_batch;
import static io.gravitee.llama.cpp.LlamaRuntime.n_ctx;
import static io.gravitee.llama.cpp.LlamaRuntime.n_seq_max;
import static io.gravitee.llama.cpp.LlamaRuntime.n_threads;
import static io.gravitee.llama.cpp.LlamaRuntime.n_threads_batch;
import static io.gravitee.llama.cpp.LlamaRuntime.n_ubatch;
import static io.gravitee.llama.cpp.LlamaRuntime.no_perf;
import static io.gravitee.llama.cpp.LlamaRuntime.offload_kqv;
import static io.gravitee.llama.cpp.LlamaRuntime.pooling_type;

import java.lang.foreign.Arena;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaContextParams extends MemorySegmentAware {

  public LlamaContextParams(Arena arena) {
    super(llama_context_default_params(arena));
  }

  public int nCtx() {
    return n_ctx(this.segment);
  }

  public LlamaContextParams nCtx(int nCtx) {
    n_ctx(this.segment, nCtx);
    return this;
  }

  public int nBatch() {
    return n_batch(this.segment);
  }

  public LlamaContextParams nBatch(int nBatch) {
    n_batch(this.segment, nBatch);
    return this;
  }

  public int nUBatch() {
    return n_ubatch(this.segment);
  }

  public LlamaContextParams nUBatch(int nUBatch) {
    n_ubatch(this.segment, nUBatch);
    return this;
  }

  public int nSeqMax() {
    return n_seq_max(this.segment);
  }

  public LlamaContextParams nSeqMax(int nSeqMax) {
    n_seq_max(this.segment, nSeqMax);
    return this;
  }

  public int nThreads() {
    return n_threads(this.segment);
  }

  public LlamaContextParams nThreads(int nThreads) {
    n_threads(this.segment, nThreads);
    return this;
  }

  public int nThreadsBatch() {
    return n_threads_batch(this.segment);
  }

  public LlamaContextParams nThreadsBatch(int nThreadsBatch) {
    n_threads_batch(this.segment, nThreadsBatch);
    return this;
  }

  public PoolingType poolingType() {
    return PoolingType.fromOrdinal(pooling_type(this.segment) + 1);
  }

  public LlamaContextParams poolingType(PoolingType poolingType) {
    pooling_type(this.segment, poolingType.ordinal() - 1);
    return this;
  }

  public AttentionType attentionType() {
    return AttentionType.fromOrdinal(attention_type(this.segment) + 1);
  }

  public LlamaContextParams attentionType(AttentionType attentionType) {
    attention_type(this.segment, attentionType.ordinal() - 1);
    return this;
  }

  public boolean embeddings() {
    return LlamaRuntime.embeddings(this.segment);
  }

  public LlamaContextParams embeddings(boolean embeddings) {
    LlamaRuntime.embeddings(this.segment, embeddings);
    return this;
  }

  public boolean offloadKQV() {
    return offload_kqv(this.segment);
  }

  public LlamaContextParams offloadKQV(boolean offloadKQV) {
    offload_kqv(this.segment, offloadKQV);
    return this;
  }

  public boolean flashAttn() {
    return flash_attn(this.segment);
  }

  public LlamaContextParams flashAttn(boolean flashAttn) {
    flash_attn(this.segment, flashAttn);
    return this;
  }

  public boolean noPerf() {
    return no_perf(this.segment);
  }

  public LlamaContextParams noPerf(boolean noPerf) {
    no_perf(this.segment, noPerf);
    return this;
  }
}
