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

    public LlamaContextParams nBatch(int nCtx){
        llama_context_params.n_batch$set(this.segment, nCtx);
        return this;
    }
}
