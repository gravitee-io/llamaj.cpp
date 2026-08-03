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
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaSampler extends MemorySegmentAware implements Freeable {

  private final SegmentAllocator allocator;

  public LlamaSampler(SegmentAllocator allocator) {
    super(
      llama_sampler_chain_init(llama_sampler_chain_default_params(allocator))
    );
    this.allocator = allocator;
  }

  public int sample(LlamaContext context) {
    return llama_sampler_sample(this.segment, context.segment, -1);
  }

  /**
   * Samples a token from the specified output index in the batch.
   * Used for parallel processing of multiple sequences.
   *
   * @param context The context containing the logits
   * @param idx The index of the output to sample from (0-based)
   * @return The sampled token ID
   */
  public int sample(LlamaContext context, int idx) {
    return llama_sampler_sample(this.segment, context.segment, idx);
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
    llama_sampler_chain_add(
      this.segment,
      llama_sampler_init_top_p(topP, minKeep)
    );
    return this;
  }

  public LlamaSampler minP(float minP, int minKeep) {
    llama_sampler_chain_add(
      this.segment,
      llama_sampler_init_min_p(minP, minKeep)
    );
    return this;
  }

  public LlamaSampler mirostat(int seed, float tau, float eta) {
    llama_sampler_chain_add(
      this.segment,
      llama_sampler_init_mirostat_v2(seed, tau, eta)
    );
    return this;
  }

  public LlamaSampler grammar(LlamaVocab vocab, String grammar, String root) {
    var grammarSegment = allocator.allocateFrom(grammar);
    var rootSegment = allocator.allocateFrom(root);
    llama_sampler_chain_add(
      this.segment,
      llama_sampler_init_grammar(vocab.segment, grammarSegment, rootSegment)
    );
    return this;
  }

  /**
   * Adds a LAZY grammar: the constraint arms only once one of {@code triggerPatterns}
   * matches the generated text, and then applies from that match onward.
   *
   * <p>This is what makes constrained tool calls possible without breaking ordinary
   * replies. {@link #grammar} forces the WHOLE response to match, so a model asked to
   * chat would have to answer in JSON. With a trigger — for Harmony, the tool-call
   * header — prose stays free and only the arguments are constrained, which is the
   * llama.cpp counterpart of vLLM's structural tags.
   *
   * <p>The grammar is fed content starting at the pattern's first capture group, so a
   * pattern normally wraps the payload it wants to constrain, e.g.
   * {@code "<\\|channel\\|>commentary to=functions\\.\\w+<\\|constrain\\|>json<\\|message\\|>(.*)"}.
   *
   * @param vocab           the model vocabulary
   * @param grammar         GBNF grammar text
   * @param root            the grammar's root rule
   * @param triggerPatterns regexes that arm the grammar; empty means never
   * @param triggerTokens   token ids that arm the grammar; may be empty
   */
  public LlamaSampler grammarLazy(
    LlamaVocab vocab,
    String grammar,
    String root,
    List<String> triggerPatterns,
    List<Integer> triggerTokens
  ) {
    var grammarSegment = allocator.allocateFrom(grammar);
    var rootSegment = allocator.allocateFrom(root);

    // const char** — an array of pointers, each to its own NUL-terminated string.
    var patterns = triggerPatterns == null
      ? List.<String>of()
      : triggerPatterns;
    MemorySegment patternArray = allocator.allocate(
      ValueLayout.ADDRESS,
      Math.max(patterns.size(), 1)
    );
    for (int i = 0; i < patterns.size(); i++) {
      patternArray.setAtIndex(
        ValueLayout.ADDRESS,
        i,
        allocator.allocateFrom(patterns.get(i))
      );
    }

    var tokens = triggerTokens == null ? List.<Integer>of() : triggerTokens;
    MemorySegment tokenArray = allocator.allocate(
      ValueLayout.JAVA_INT,
      Math.max(tokens.size(), 1)
    );
    for (int i = 0; i < tokens.size(); i++) {
      tokenArray.setAtIndex(ValueLayout.JAVA_INT, i, tokens.get(i));
    }

    llama_sampler_chain_add(
      this.segment,
      llama_sampler_init_grammar_lazy_patterns(
        vocab.segment,
        grammarSegment,
        rootSegment,
        patternArray,
        patterns.size(),
        tokenArray,
        tokens.size()
      )
    );
    return this;
  }

  public LlamaSampler penalties(
    int penaltyLastN,
    float penaltyRepeat,
    float penaltyFreq,
    float penaltyPresent
  ) {
    llama_sampler_chain_add(
      this.segment,
      llama_sampler_init_penalties(
        penaltyLastN,
        penaltyRepeat,
        penaltyFreq,
        penaltyPresent
      )
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
