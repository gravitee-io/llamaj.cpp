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

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaVocab extends MemorySegmentAware {

  private static final int BUFFER_SIZE = 256;

  public LlamaVocab(LlamaModel model) {
    super(llama_model_get_vocab(model.segment));
  }

  public boolean isEog(int tokenId) {
    return llama_vocab_is_eog(this.segment, tokenId);
  }

  public byte[] tokenToPiece(int tokenId) {
    int bufferSize = BUFFER_SIZE;
    try (Arena arena = Arena.ofConfined()) {
      // This loop handles the case where the initial buffer is too small for the token piece.
      // The native llama_token_to_piece function will tell us the required size.
      while (true) {
        var buffer = arena.allocateArray(ValueLayout.JAVA_BYTE, bufferSize);
        int pieceLength = llama_token_to_piece(this.segment, tokenId, buffer, (int) buffer.byteSize(), 0, true);

        // If pieceLength is negative, the buffer was too small. The absolute value
        // indicates the required buffer size. We resize and try again.
        if (pieceLength < 0) {
          bufferSize = Math.max(bufferSize * 2, -pieceLength);
          continue;
        }

        // If pieceLength is 0, it's an empty token.
        if (pieceLength == 0) {
          return new byte[0];
        }

        // If the piece fits in the buffer, copy it to a new array of the exact size.
        if (pieceLength <= bufferSize) {
          byte[] bytes = buffer.toArray(ValueLayout.JAVA_BYTE);
          byte[] out = new byte[pieceLength];
          System.arraycopy(bytes, 0, out, 0, pieceLength);
          return out;
        }

        // This case should ideally not be hit if the negative length logic is correct,
        // but as a safeguard, if the reported length is larger than our buffer,
        // we resize the buffer and retry.
        bufferSize = Math.max(bufferSize * 2, pieceLength);
      }
    }
  }
}
