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
import java.lang.foreign.ValueLayout;

import static io.gravitee.llama.cpp.LlamaRuntime.llama_chat_apply_template;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_model_chat_template;


/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaTemplate extends MemorySegmentAware {

    public LlamaTemplate(LlamaModel llamaModel) {
        super(llama_model_chat_template(llamaModel.segment, MemorySegment.NULL));
    }

    public String applyTemplate(SegmentAllocator allocator, LlamaChatMessages messages, int nCtx) {
        var templateBuffer = allocator.allocateArray(ValueLayout.JAVA_CHAR, nCtx);
        
        int newLength = llama_chat_apply_template(
                segment,
                messages.segment,
                messages.getMessages().size(),
                true,
                templateBuffer,
                nCtx
        );

        if (newLength > nCtx) {
            templateBuffer = allocator.allocateArray(ValueLayout.JAVA_CHAR, newLength);
            newLength = llama_chat_apply_template(
                    segment,
                    messages.segment,
                    messages.getMessages().size(),
                    true,
                    templateBuffer,
                    newLength
            );
        }

        if (newLength < 0) {
            throw new IllegalStateException("failed to apply the chat template.");
        }

        return templateBuffer.getUtf8String(0);
    }
}
