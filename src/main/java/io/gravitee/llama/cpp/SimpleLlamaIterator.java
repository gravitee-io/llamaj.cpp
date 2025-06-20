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

import static java.util.function.Predicate.not;

import io.gravitee.llama.cpp.LlamaTokenizer.TokenizerResponse;
import java.lang.foreign.Arena;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class SimpleLlamaIterator extends LlamaIterator {

  private TokenizerResponse tokenized;

  private final LlamaContext context;
  private final LlamaSampler sampler;

  private final AtomicInteger inputTokens = new AtomicInteger(0);
  private final AtomicInteger outputTokens = new AtomicInteger(0);

  private LlamaBatch batch;
  private Integer newTokenId;
  private int quota = -1;
  private boolean hasNext;

  private List<String> stopStrings = List.of();
  private String promptMemory = "";
  private int maxStopStringSize = 0;

  private final LlamaTokenizer tokenizer;

  public SimpleLlamaIterator(Arena arena, LlamaContext context, LlamaTokenizer tokenizer, LlamaSampler sampler) {
    super(arena);
    this.context = context;
    this.sampler = sampler;
    this.tokenizer = tokenizer;
  }

  public SimpleLlamaIterator initialize(String prompt) {
    tokenized = tokenizer.tokenize(arena, prompt);
    inputTokens.set(tokenized.size());
    hasNext = batch();
    return this;
  }

  @Override
  public boolean hasNext() {
    return hasNext;
  }

  private boolean batch() {
    batch = newTokenId == null ? new LlamaBatch(arena, tokenized) : new LlamaBatch(arena, newTokenId);

    if (checkContextSize() && batch.decode(context) != 0) {
      return false;
    }

    newTokenId = sampler.sample(context);

    batch.free();
    batch = null;

    outputTokens.incrementAndGet();
    return hasNotReachedQuota() && !tokenizer.isEog(newTokenId);
  }

  private boolean hasNotReachedQuota() {
    return quota == -1 || quota > outputTokens.get();
  }

  private boolean checkContextSize() {
    return context.nCtxUsedCells() + batch.nTokens() <= context.nCtx();
  }

  @Override
  public LlamaOutput next() {
    var piece = tokenizer.tokenToPiece(newTokenId);

    if (!stopStrings.isEmpty()) {
      promptMemory += piece;

      if (promptMemory.length() > maxStopStringSize) {
        promptMemory = promptMemory.substring(promptMemory.length() - maxStopStringSize);
      }
    }

    hasNext = stopStringNotEndsWith() && batch();
    return new LlamaOutput(piece, 1);
  }

  public SimpleLlamaIterator setQuota(int quota) {
    this.quota = Math.min(quota, context.nCtx());
    return this;
  }

  public SimpleLlamaIterator setStopStrings(List<String> stopStrings) {
    this.stopStrings = stopStrings.stream().filter(not(String::isBlank)).toList();
    maxStopStringSize = this.stopStrings.stream().mapToInt(String::length).max().orElse(0);
    return this;
  }

  private boolean stopStringNotEndsWith() {
    return (stopStrings.isEmpty() || stopStrings.stream().noneMatch(promptMemory::endsWith));
  }

  public int getInputTokens() {
    return inputTokens.get();
  }

  public int getOutputTokens() {
    return outputTokens.get();
  }
}
