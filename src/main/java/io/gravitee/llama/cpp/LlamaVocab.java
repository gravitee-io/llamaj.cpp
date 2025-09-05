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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_model_get_vocab;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_token_to_piece;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_vocab_is_eog;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaVocab extends MemorySegmentAware {

  public LlamaVocab(LlamaModel model) {
    super(llama_model_get_vocab(model.segment));
  }

  public boolean isEog(int tokenId) {
    return llama_vocab_is_eog(this.segment, tokenId);
  }

  public String tokenToPiece(int tokenId) {
    try (Arena arena = Arena.ofConfined()) {
      var buffer = arena.allocateArray(ValueLayout.JAVA_BYTE, 256);

      int pieceLength = llama_token_to_piece(this.segment, tokenId, buffer, (int) buffer.byteSize(), 0, true);

      if (pieceLength <= 0) {
        return "";
      }

      byte[] bytes = buffer.toArray(ValueLayout.JAVA_BYTE);
      return new String(bytes, 0, pieceLength, StandardCharsets.UTF_8);
    }
  }
}
