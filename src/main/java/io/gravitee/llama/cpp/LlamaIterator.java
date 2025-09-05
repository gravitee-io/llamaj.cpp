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
import static java.util.function.Predicate.not;

import io.gravitee.llama.cpp.LlamaTokenizer.TokenizerResponse;
import java.lang.foreign.Arena;
import java.util.Iterator;
import java.util.List;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.atomic.AtomicInteger;
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

  protected final AtomicInteger inputTokens = new AtomicInteger(0);
  protected final AtomicInteger outputTokens = new AtomicInteger(0);

  private List<String> stopStrings = List.of();
  private String promptMemory = "";
  protected int maxStopStringSize = 0;

  private FinishReason finishReason;

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
    inputTokens.set(tokenized.size());
    hasNext = batch();
    return this;
  }

  public abstract boolean batch();

  protected boolean isEog(int tokenId) {
    boolean isEog = tokenizer.isEog(tokenId);
    if (isEog) {
      finishReason = EOS;
    }
    return isEog;
  }

  protected boolean hasNotReachedQuota() {
    boolean hasNotReachedQuota = maxTokens == -1 || maxTokens > outputTokens.get();
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
    return inputTokens.get();
  }

  public int getOutputTokens() {
    return outputTokens.get();
  }

  public FinishReason getFinishReason() {
    return finishReason;
  }

  public LlamaIterator setStopStrings(List<String> stopStrings) {
    this.stopStrings = stopStrings.stream().filter(not(String::isBlank)).toList();
    maxStopStringSize = this.stopStrings.stream().mapToInt(String::length).max().orElse(0);
    return this;
  }

  protected boolean endWithStopString() {
    boolean endsWithStopString = !(stopStrings.isEmpty() || stopStrings.stream().noneMatch(promptMemory::endsWith));
    if (endsWithStopString) {
      finishReason = FinishReason.STOP;
    }
    return endsWithStopString;
  }

  protected void feedPromptMemory(String piece) {
    if (!stopStrings.isEmpty()) {
      promptMemory += piece;

      if (promptMemory.length() > maxStopStringSize) {
        promptMemory = promptMemory.substring(promptMemory.length() - maxStopStringSize);
      }
    }
  }
}
