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
import java.nio.file.Path;

import static io.gravitee.llama.cpp.LlamaRuntime.llama_model_load_from_file;


/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaModel extends MemorySegmentAware implements Freeable {

    public LlamaModel(SegmentAllocator arena, Path modelPath, LlamaModelParams params) {
        this(llama_model_load_from_file(getModelAsString(arena, modelPath), params.segment));
    }

    public LlamaModel(MemorySegment segment) {
        super(segment);
    }

    private static MemorySegment getModelAsString(SegmentAllocator arena, Path modelPath) {
        return arena.allocateUtf8String(modelPath.toAbsolutePath().toString());
    }

    @Override
    public void free() {
        LlamaRuntime.llama_model_free(this);
    }
}
