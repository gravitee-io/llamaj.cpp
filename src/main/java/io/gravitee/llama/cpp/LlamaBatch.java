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

import static io.gravitee.llama.cpp.LlamaRuntime.*;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import io.gravitee.llama.cpp.LlamaTokenizer.TokenizerResponse;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaBatch extends MemorySegmentAware implements Freeable {

  public LlamaBatch(SegmentAllocator allocator, TokenizerResponse tokenizerResponse) {
    this(allocator, tokenizerResponse.data(), tokenizerResponse.size());
  }

  public LlamaBatch(SegmentAllocator allocator, int tokenId) {
    this(allocator, getTokenArray(allocator, tokenId), 1);
  }

  public LlamaBatch(SegmentAllocator allocator, MemorySegment segment, int size) {
    super(llama_batch_get_one(allocator, segment, size));
  }

  private static MemorySegment getTokenArray(SegmentAllocator allocator, int tokenId) {
    var tokenArray = allocator.allocateArray(JAVA_INT, 1);
    tokenArray.set(JAVA_INT, 0, tokenId);
    return tokenArray;
  }

  public int decode(LlamaContext context) {
    return llama_decode(context.segment, this.segment);
  }

  public int nTokens() {
    return llama_batch_n_tokens(segment);
  }

  @Override
  public void free() {
    checkNotFreed();
    markFreed();
    llama_batch_free(this);
  }
}
