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

import static io.gravitee.llama.cpp.llama_h_1.llama_get_kv_cache_used_cells;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaIterator extends ArenaAware implements Iterator<String> {

    private final TokenizerResponse tokenized;

    private final LlamaContext context;
    private final LlamaVocab vocab;
    private final LlamaSampler sampler;
    private final int nCtx;

    private LlamaBatch batch;
    private Integer newTokenId;

    private boolean hasNext;

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
        return !vocab.isEog(newTokenId);
    }

    private boolean checkContextSize() {
        int nCtxUsed = llama_get_kv_cache_used_cells(context.segment);
        return nCtxUsed + batch.nTokens() <= nCtx;
    }

    @Override
    public String next() {
        String s = vocab.tokenToPiece(arena, newTokenId);
        hasNext = batch();
        return s;
    }
}
