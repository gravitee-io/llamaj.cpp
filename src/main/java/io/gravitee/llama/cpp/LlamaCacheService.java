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

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
final class LlamaCacheService extends ArenaAware {

    private final HashMap<UUID, LlamaCacheView> cache;

    LlamaCacheService(Arena arena) {
        super(arena);
        this.cache = new HashMap<>();
    }

    public UUID newCache(LlamaContext context) {
        UUID key = UUID.randomUUID();
        this.cache.put(key, new LlamaCacheView(arena, context));
        return key;
    }

    public LlamaCacheView get(UUID key) {
        return cache.get(key);
    }

    public void remove(UUID key) {
        LlamaCacheView cacheView = cache.remove(key);
        if (cacheView != null) {
            cacheView.free();
        }
    }

}
