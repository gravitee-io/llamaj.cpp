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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static io.gravitee.llama.cpp.llama_h_1.*;

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
        return (quota == -1 || quota > outputTokens.incrementAndGet()) && !vocab.isEog(newTokenId);
    }

    private boolean checkContextSize() {
        int nCtxUsed = llama_get_kv_cache_used_cells(context.segment);
        return nCtxUsed + batch.nTokens() <= nCtx;
    }

    @Override
    public LlamaOutput next() {
        String s = vocab.tokenToPiece(arena, newTokenId);
        hasNext = batch();
        return new LlamaOutput(s, 1);
    }

    public LlamaIterator setQuota(int quota) {
        this.quota = quota;
        return this;
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
