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

import static io.gravitee.llama.cpp.llama_h_1.llama_context_default_params;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaContextParams extends MemorySegmentAware {

    public LlamaContextParams(Arena arena) {
        super(llama_context_default_params(arena));
    }

    public int nCtx(){
        return llama_context_params.n_ctx$get(this.segment);
    }

    public LlamaContextParams nCtx(int nCtx){
        llama_context_params.n_ctx$set(this.segment, nCtx);
        return this;
    }

    public int nBatch(){
        return llama_context_params.n_batch$get(this.segment);
    }

    public LlamaContextParams nBatch(int nBatch){
        llama_context_params.n_batch$set(this.segment, nBatch);
        return this;
    }

    public int nUBatch(){
        return llama_context_params.n_ubatch$get(this.segment);
    }

    public LlamaContextParams nUBatch(int nUBatch){
        llama_context_params.n_ubatch$set(this.segment, nUBatch);
        return this;
    }

    public int nSeqMax(){
        return llama_context_params.n_seq_max$get(this.segment);
    }

    public LlamaContextParams nSeqMax(int nSeqMax){
        llama_context_params.n_seq_max$set(this.segment, nSeqMax);
        return this;
    }

    public int nThreads(){
        return llama_context_params.n_threads$get(this.segment);
    }

    public LlamaContextParams nThreads(int nThreads){
        llama_context_params.n_threads$set(this.segment, nThreads);
        return this;
    }

    public int nThreadsBatch(){
        return llama_context_params.n_threads_batch$get(this.segment);
    }

    public LlamaContextParams nThreadsBatch(int nThreadsBatch){
        llama_context_params.n_threads_batch$set(this.segment, nThreadsBatch);
        return this;
    }

    public PoolingType poolingType(){
        return PoolingType.fromOrdinal(llama_context_params.pooling_type$get(this.segment) + 1);
    }

    public LlamaContextParams poolingType(PoolingType poolingType){
        llama_context_params.pooling_type$set(this.segment, poolingType.ordinal() - 1);
        return this;
    }

    public AttentionType attentionType(){
        return AttentionType.fromOrdinal(llama_context_params.attention_type$get(this.segment) + 1);
    }

    public LlamaContextParams attentionType(AttentionType attentionType){
        llama_context_params.attention_type$set(this.segment, attentionType.ordinal() - 1);
        return this;
    }


    public boolean embeddings(){
        return llama_context_params.embeddings$get(this.segment);
    }

    public LlamaContextParams embeddings(boolean embeddings) {
        llama_context_params.embeddings$set(this.segment, embeddings);
        return this;
    }

    public boolean offloadKQV(){
        return llama_context_params.offload_kqv$get(this.segment);
    }

    public LlamaContextParams offloadKQV(boolean offloadKQV) {
        llama_context_params.offload_kqv$set(this.segment, offloadKQV);
        return this;
    }

    public boolean flashAttn(){
        return llama_context_params.flash_attn$get(this.segment);
    }

    public LlamaContextParams flashAttn(boolean flashAttn) {
        llama_context_params.flash_attn$set(this.segment, flashAttn);
        return this;
    }

    public boolean noPerf(){
        return llama_context_params.no_perf$get(this.segment);
    }

    public LlamaContextParams noPerf(boolean embeddings) {
        llama_context_params.no_perf$set(this.segment, embeddings);
        return this;
    }
}
