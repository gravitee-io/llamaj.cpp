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

import static io.gravitee.llama.cpp.FinishReason.EOS;

import io.gravitee.llama.cpp.LlamaTokenizer.TokenizerResponse;
import io.gravitee.llama.cpp.modules.PromptMemory;
import io.gravitee.llama.cpp.modules.StopString;
import io.gravitee.llama.cpp.modules.TokenTracking;
import java.lang.foreign.Arena;
import java.util.Iterator;
import java.util.List;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public abstract class LlamaIterator extends ArenaAware implements Iterator<LlamaOutput> {

  protected final LlamaContext context;
  protected final LlamaSampler sampler;
  protected final LlamaTokenizer tokenizer;

  protected int maxTokens = -1;
  protected TokenizerResponse tokenized;
  protected boolean hasNext;

  protected final TokenTracking tokenTracking = new TokenTracking();
  protected final PromptMemory promptMemory = new PromptMemory();
  protected final StopString stopString = new StopString();

  private FinishReason finishReason;
  private GenerationState state;

  public LlamaIterator(Arena allocator, LlamaContext context, LlamaTokenizer tokenizer, LlamaSampler sampler) {
    super(allocator);
    this.context = context;
    this.sampler = sampler;
    this.tokenizer = tokenizer;
  }

  public Stream<LlamaOutput> stream() {
    return StreamSupport.stream(Spliterators.spliteratorUnknownSize(this, Spliterator.ORDERED), false);
  }

  public LlamaIterator initialize(String prompt) {
    tokenized = tokenizer.tokenize(arena, prompt);
    tokenTracking.initialize(tokenized.size());
    state = GenerationState.OUTPUT;
    hasNext = batch();
    return this;
  }

  public abstract boolean batch();

  public void incrementTokenCount(int tokenCount) {
    tokenTracking.consume(new TokenTracking.Context(state, tokenCount));
  }

  protected boolean isEog(int tokenId) {
    boolean isEog = tokenizer.isEog(tokenId);
    if (isEog) {
      finishReason = EOS;
    }
    return isEog;
  }

  protected boolean hasNotReachedQuota() {
    boolean hasNotReachedQuota = maxTokens == -1 || maxTokens > getOutputTokens();
    if (!hasNotReachedQuota) {
      finishReason = FinishReason.LENGTH;
    }
    return hasNotReachedQuota;
  }

  public LlamaIterator setMaxTokens(int maxTokens) {
    this.maxTokens = Math.min(maxTokens, context.nCtx());
    return this;
  }

  public int getInputTokens() {
    return tokenTracking.getInputTokenCount();
  }

  public int getOutputTokens() {
    return tokenTracking.getOutputTokenCount();
  }

  public FinishReason getFinishReason() {
    return finishReason;
  }

  public LlamaIterator setStopStrings(List<String> stopStrings) {
    stopString.initialize(stopStrings);
    int maxStringSize = stopStrings.stream().mapToInt(String::length).max().orElse(0);
    promptMemory.initialize(maxStringSize);
    return this;
  }

  protected boolean endWithStopString() {
    if (!promptMemory.isInitialized()) {
      return false;
    }

    boolean endsWithStopString = stopString.evaluate(promptMemory.getMemory());
    if (endsWithStopString) {
      finishReason = FinishReason.STOP;
    }
    return endsWithStopString;
  }

  protected void feedPromptMemory(String tokenPiece) {
    if (promptMemory.isInitialized()) {
      promptMemory.consume(tokenPiece);
    }
  }
}
