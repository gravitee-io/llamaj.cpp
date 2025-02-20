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

import io.gravitee.llama.cpp.LlamaTokenizer.TokenizerResponse;

import java.lang.foreign.Arena;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;

import static io.gravitee.llama.cpp.llama_h_1.*;
import static java.util.function.Predicate.not;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaIterator extends ArenaAware implements Iterator<LlamaOutput> {

    private final TokenizerResponse tokenized;

    private final LlamaContext context;
    private final LlamaVocab vocab;
    private final LlamaSampler sampler;
    private final int nCtx;

    private final AtomicInteger inputTokens = new AtomicInteger(0);
    private final AtomicInteger outputTokens = new AtomicInteger(0);

    private LlamaBatch batch;
    private Integer newTokenId;
    private int quota = -1;
    private boolean hasNext;

    private List<String> stopStrings = List.of();
    private String promptMemory = "";
    private int maxStopStringSize = 0;

    public LlamaIterator(
            LlamaContext context,
            LlamaVocab vocab,
            LlamaSampler sampler,
            String prompt) {
        this(context, vocab, sampler, prompt, llama_n_ctx(context.segment));
    }

    public LlamaIterator(
            LlamaContext context,
            LlamaVocab vocab,
            LlamaSampler sampler,
            String prompt,
            int nCtx) {
        super(Arena.ofAuto());

        this.context = context;
        this.vocab = vocab;
        this.sampler = sampler;
        this.nCtx = nCtx;
        tokenized = new LlamaTokenizer(this.vocab, this.context).tokenize(arena, prompt);

        inputTokens.set(tokenized.size());

        hasNext = batch();
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

        llama_batch_free(batch.segment);
        batch = null;
        outputTokens.incrementAndGet();
        return hasNotReachedQuota() && !vocab.isEog(newTokenId);
    }

    private boolean hasNotReachedQuota() {
        return quota == -1 || quota > outputTokens.get();
    }

    private boolean checkContextSize() {
        int nCtxUsed = llama_get_kv_cache_used_cells(context.segment);
        return nCtxUsed + batch.nTokens() <= nCtx;
    }

    @Override
    public LlamaOutput next() {
        var piece = vocab.tokenToPiece(arena, newTokenId);

        if (!stopStrings.isEmpty()) {
            promptMemory += piece;

            if (promptMemory.length() > maxStopStringSize) {
                promptMemory = promptMemory.substring(promptMemory.length() - maxStopStringSize);
            }
        }

        hasNext = stopStringNotEndsWith() && batch();
        return new LlamaOutput(piece, 1);
    }

    public LlamaIterator setQuota(int quota) {
        this.quota = quota;
        return this;
    }

    public LlamaIterator setStopStrings(List<String> stopStrings) {
        this.stopStrings = stopStrings.stream().filter(not(String::isBlank)).toList();
        maxStopStringSize = this.stopStrings.stream().mapToInt(String::length).max().orElse(0);
        return this;
    }

    private boolean stopStringNotEndsWith() {
        return stopStrings.isEmpty() || stopStrings.stream().noneMatch(promptMemory::endsWith);
    }

    @Override
    public void close() {
        super.close();
    }

    public int getInputTokens() {
        return inputTokens.get();
    }

    public int getOutputTokens() {
        return outputTokens.get();
    }
}
