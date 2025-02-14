
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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaChatMessages extends MemorySegmentAware {

    private final List<LlamaChatMessage> messages;

    public LlamaChatMessages(Arena arena, List<LlamaChatMessage> messages) {
        super(initMessages(arena, messages));
        this.messages = messages;
    }

    private static MemorySegment initMessages(Arena arena, List<LlamaChatMessage> messages) {
        long structSize = llama_chat_message.sizeof();
        var chatArray = llama_chat_message.allocateArray(messages.size(), arena);
        for (var index = 0; index < messages.size(); index++) {
            var messageSegment = messages.get(index).segment;
            long structOffset = index * structSize;
            chatArray.asSlice(structOffset, structSize).copyFrom(messageSegment);
        }
        return chatArray;
    }

    public List<LlamaChatMessage> getMessages() {
        return messages;
    }
}
