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

import java.lang.foreign.*;
import java.util.function.Consumer;

import static io.gravitee.llama.cpp.llama_h_1.llama_log_set;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaLogger extends ArenaAware {

    private LlamaLogLevel level;
    private Consumer<String> logger;

    public LlamaLogger(Arena arena) {
        super(arena);
    }

    private void logCallback(int level, MemorySegment text, MemorySegment user_data) {
        if (this.level.ordinal() <= level) {
            this.logger.accept(text.getUtf8String(0));
        }
    }

    public void setLogging(LlamaLogLevel level, Consumer<String> logger) {
        this.level = level;
        this.logger = logger;

        ggml_log_callback logCallback = this::logCallback;
        llama_log_set(ggml_log_callback.allocate(logCallback, arena), MemorySegment.NULL);
    }

    public void setLogging(LlamaLogLevel level) {
        this.setLogging(level, s -> System.out.println(s.replace("\n", "")));
    }
}
