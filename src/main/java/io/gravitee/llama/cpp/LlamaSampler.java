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
import java.lang.foreign.SegmentAllocator;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaSampler extends MemorySegmentAware implements Freeable {

  private final SegmentAllocator allocator;

  public LlamaSampler(SegmentAllocator allocator) {
    super(llama_sampler_chain_init(llama_sampler_chain_default_params(allocator)));
    this.allocator = allocator;
  }

  public int sample(LlamaContext context) {
    return llama_sampler_sample(this.segment, context.segment, -1);
  }

  public LlamaSampler greedy() {
    llama_sampler_chain_add(this.segment, llama_sampler_init_greedy());
    return this;
  }

  public LlamaSampler temperature(float temperature) {
    llama_sampler_chain_add(this.segment, llama_sampler_init_temp(temperature));
    return this;
  }

  public LlamaSampler topK(int topK) {
    llama_sampler_chain_add(this.segment, llama_sampler_init_top_k(topK));
    return this;
  }

  public LlamaSampler topP(float topP, int minKeep) {
    llama_sampler_chain_add(this.segment, llama_sampler_init_top_p(topP, minKeep));
    return this;
  }

  public LlamaSampler minP(float minP, int minKeep) {
    llama_sampler_chain_add(this.segment, llama_sampler_init_min_p(minP, minKeep));
    return this;
  }

  public LlamaSampler mirostat(int seed, float tau, float eta) {
    llama_sampler_chain_add(this.segment, llama_sampler_init_mirostat_v2(seed, tau, eta));
    return this;
  }

  public LlamaSampler grammar(LlamaVocab vocab, String grammar, String root) {
    var grammarSegment = allocator.allocateUtf8String(grammar);
    var rootSegment = allocator.allocateUtf8String(root);
    llama_sampler_chain_add(this.segment, llama_sampler_init_grammar(vocab.segment, grammarSegment, rootSegment));
    return this;
  }

  public LlamaSampler penalties(int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent) {
    llama_sampler_chain_add(
      this.segment,
      llama_sampler_init_penalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent)
    );
    return this;
  }

  public LlamaSampler seed(int seed) {
    llama_sampler_chain_add(this.segment, llama_sampler_init_dist(seed));
    return this;
  }

  public LlamaPerformance.SamplerPerformance getPerformance(Arena arena) {
    checkNotFreed();
    MemorySegment perfData = llama_perf_sampler(arena, segment);
    return new LlamaPerformance.SamplerPerformance(
      llama_perf_sampler_t_sample_ms(perfData),
      llama_perf_sampler_n_sample(perfData)
    );
  }

  @Override
  public void free() {
    checkNotFreed();
    markFreed();
    llama_sampler_free(this);
  }
}
