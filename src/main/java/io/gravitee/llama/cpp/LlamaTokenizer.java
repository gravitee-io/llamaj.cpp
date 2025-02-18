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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;

import static io.gravitee.llama.cpp.llama_h_1.*;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaTokenizer extends MemorySegmentAware {

    private final LlamaContext context;
    private final LlamaVocab vocab;

    public LlamaTokenizer(LlamaVocab vocab, LlamaContext llamaContext) {
        super(null);
        this.vocab = vocab;
        this.context = llamaContext;
    }

    public TokenizerResponse tokenize(SegmentAllocator allocator, String prompt) {
        boolean isFirst = llama_get_kv_cache_used_cells(context.segment) == 0;
        var promptSegment = allocator.allocateUtf8String(prompt);
        int nbPromptTokens = -llama_tokenize(
                vocab.segment,
                promptSegment,
                prompt.length(),
                MemorySegment.NULL,
                0,
                isFirst,
                true
        );

        var tokenBuffer = allocator.allocateArray(llama_token, nbPromptTokens);

        if (llama_tokenize(vocab.segment, promptSegment, prompt.length(), tokenBuffer, nbPromptTokens, isFirst, true) < 0) {
            throw new IllegalStateException("Failed to tokenize");
        }

        var tokenizerResponse = new TokenizerResponse(tokenBuffer, nbPromptTokens);
        llama_kv_cache_clear(context.segment);
        return tokenizerResponse;
    }

    public record TokenizerResponse(MemorySegment data, int size){}
}
