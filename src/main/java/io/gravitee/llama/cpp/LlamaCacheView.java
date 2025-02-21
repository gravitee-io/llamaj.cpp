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
import java.util.HashMap;
import java.util.UUID;

import static io.gravitee.llama.cpp.macosx.aarch64.llama_h_1.llama_kv_cache_view_free;
import static io.gravitee.llama.cpp.macosx.aarch64.llama_h_1.llama_kv_cache_view_init;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
final class LlamaCacheView extends MemorySegmentAware {

    LlamaCacheView(Arena arena, LlamaContext context) {
        super(llama_kv_cache_view_init(arena, context.segment, context.nCtx()));
    }

    public void free() {
        llama_kv_cache_view_free(segment);
    }
}
