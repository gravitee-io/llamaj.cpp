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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_tokenize;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaTokenizer {

  private final LlamaVocab vocab;
  private final LlamaContext context;

  public LlamaTokenizer(LlamaVocab vocab, LlamaContext context) {
    this.vocab = vocab;
    this.context = context;
  }

  public TokenizerResponse tokenize(SegmentAllocator allocator, String prompt) {
    boolean isFirst = this.context.nCtxUsedCells() == 0;

    var promptSegment = this.getPromptSegment(allocator, prompt, isFirst);

    int nbPromptTokens = promptSegment.size();
    var tokenBuffer = allocator.allocateArray(JAVA_INT, nbPromptTokens);

    if (llama_tokenize(vocab.segment, promptSegment.data, prompt.length(), tokenBuffer, nbPromptTokens, isFirst, true) < 0) {
      throw new IllegalStateException("Failed to tokenize");
    }

    return new TokenizerResponse(tokenBuffer, nbPromptTokens);
  }

  public PromptSegment getPromptSegment(SegmentAllocator allocator, String prompt, boolean isFirst) {
    var promptSegment = allocator.allocateUtf8String(prompt);
    int nbPromptTokens = -llama_tokenize(
      vocab.segment,
      promptSegment,
      prompt.length(),
      MemorySegment.NULL,
      0,
      isFirst,
      true
    );

    return new PromptSegment(promptSegment, nbPromptTokens);
  }

  public boolean isEog(int tokenId) {
    return vocab.isEog(tokenId);
  }

  public byte[] tokenToPiece(int tokenId) {
    return vocab.tokenToPiece(tokenId);
  }

  public record PromptSegment(MemorySegment data, int size) {}

  public record TokenizerResponse(MemorySegment data, int size) {}
}
