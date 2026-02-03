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

import static io.gravitee.llama.cpp.LlamaRuntime.mtmd_input_chunk_get_tokens_text;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Represents a single chunk of multimodal input (text or image).
 * Wraps the native `mtmd_input_chunk` structure.
 */
public class MtmdInputChunk {

  private final MemorySegment chunkSegment;

  public MtmdInputChunk(MemorySegment chunkSegment) {
    this.chunkSegment = chunkSegment;
  }

  public MemorySegment getChunkSegment() {
    return chunkSegment;
  }

  public MtmdInputChunkType getType() {
    int type = LlamaRuntime.mtmd_input_chunk_get_type(chunkSegment);
    return MtmdInputChunkType.fromOrdinal(type);
  }

  public MemorySegment getTextTokens(Arena arena) {
    // Modified line
    MemorySegment nTokensOutput = arena.allocate(ValueLayout.JAVA_LONG); // New line: allocate MemorySegment for output
    return mtmd_input_chunk_get_tokens_text(
      chunkSegment,
      nTokensOutput // Changed from MemorySegment.NULL to nTokensOutput
    );
  }

  public MemorySegment getImageTokens() {
    return LlamaRuntime.mtmd_input_chunk_get_tokens_image(chunkSegment);
  }

  public long nTokens() {
    return LlamaRuntime.mtmd_input_chunk_get_n_tokens(chunkSegment);
  }

  public String getId() {
    MemorySegment idSegment = LlamaRuntime.mtmd_input_chunk_get_id(
      chunkSegment
    );
    return idSegment != MemorySegment.NULL ? idSegment.getString(0) : null;
  }

  public long nPos() {
    return LlamaRuntime.mtmd_input_chunk_get_n_pos(chunkSegment);
  }
}
