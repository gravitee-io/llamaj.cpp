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
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Represents a list of multimodal input chunks.
 * Wraps the native `mtmd_input_chunks` structure and provides an iterator.
 */
public class MtmdInputChunks implements Freeable, Iterable<MtmdInputChunk> {

  private MemorySegment chunksSegment;

  public MtmdInputChunks(MemorySegment chunksSegment) {
    this.chunksSegment = chunksSegment;
  }

  public long size() {
    return LlamaRuntime.mtmd_input_chunks_size(chunksSegment);
  }

  public MemorySegment segment() {
    return chunksSegment;
  }

  public MtmdInputChunk get(long index) {
    if (index < 0 || index >= size()) {
      throw new IndexOutOfBoundsException(
        "Index: " + index + ", Size: " + size()
      );
    }
    MemorySegment chunkSegment = LlamaRuntime.mtmd_input_chunks_get(
      chunksSegment,
      index
    );
    return new MtmdInputChunk(chunkSegment);
  }

  @Override
  public void free() {
    if (chunksSegment != null && chunksSegment.address() != 0) {
      LlamaRuntime.mtmd_input_chunks_free(chunksSegment);
      chunksSegment = null;
    }
  }

  @Override
  public boolean isFree() {
    return chunksSegment == null || chunksSegment.address() == 0;
  }

  @Override
  public Iterator<MtmdInputChunk> iterator() {
    return new Iterator<>() {
      private int currentIndex = 0;
      private final long totalSize = size();

      @Override
      public boolean hasNext() {
        return currentIndex < totalSize;
      }

      @Override
      public MtmdInputChunk next() {
        if (!hasNext()) {
          throw new NoSuchElementException();
        }
        return get(currentIndex++);
      }
    };
  }
}
