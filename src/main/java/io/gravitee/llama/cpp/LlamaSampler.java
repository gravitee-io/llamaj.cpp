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

import static io.gravitee.llama.cpp.llama_h_1.*;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaSampler extends MemorySegmentAware {

    public LlamaSampler(Arena arena) {
        super(llama_sampler_chain_init(llama_sampler_chain_default_params(arena)));
    }

    public int sample(LlamaContext context) {
        return llama_sampler_sample(this.segment, context.segment, -1);
    }

    public LlamaSampler minP(float minP) {
        llama_sampler_chain_add(this.segment, llama_sampler_init_min_p(minP, 1));
        return this;
    }

    public LlamaSampler temperature(float temperature) {
        llama_sampler_chain_add(this.segment, llama_sampler_init_temp(temperature));
        return this;
    }

    public LlamaSampler seed(int seed) {
        llama_sampler_chain_add(this.segment, llama_sampler_init_dist(seed));
        return this;
    }
}
