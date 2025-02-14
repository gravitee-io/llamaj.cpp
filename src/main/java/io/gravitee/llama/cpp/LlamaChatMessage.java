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

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaChatMessage extends MemorySegmentAware {

    public LlamaChatMessage(SegmentAllocator arena, Role role, String content) {
        super(initMessage(arena, role, content));
    }

    private static MemorySegment initMessage(SegmentAllocator allocator, Role role, String content) {
        var llamaChatMessage = llama_chat_message.allocate(allocator);
        llama_chat_message.content$set(llamaChatMessage, allocator.allocateUtf8String(content));
        llama_chat_message.role$set(llamaChatMessage, allocator.allocateUtf8String(role.getLabel()));
        return llamaChatMessage;
    }

    public Role getRole() {
        return Role.fromLabel(llama_chat_message.role$get(this.segment).getUtf8String(0));
    }

    public String getContent() {
        return llama_chat_message.content$get(this.segment).getUtf8String(0);
    }

}
