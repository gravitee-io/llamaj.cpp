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
import static io.gravitee.llama.cpp.LlamaRuntime.llama_n_vocab;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_token_to_piece;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_vocab_bos;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_vocab_eos;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_vocab_get_text;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_vocab_is_eog;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaVocab extends MemorySegmentAware {

  private static final int BUFFER_SIZE = 256;

  /** Lazily scanned once; see {@link #eogTokens()}. */
  private volatile int[] eogTokens;

  public LlamaVocab(LlamaModel model) {
    super(llama_model_get_vocab(model.segment));
  }

  public boolean isEog(int tokenId) {
    return llama_vocab_is_eog(this.segment, tokenId);
  }

  public int nVocab() {
    return llama_n_vocab(this.segment);
  }

  /**
   * Every end-of-generation token id in this vocabulary, ascending.
   *
   * <p>A vocabulary usually has several — EOS, EOT, and chat-format terminators such as
   * {@code <|im_end|>}. Callers that treat only EOS as "the" stop token leave the model an
   * unbiased alternative, which shows up as a weaker effect rather than a failure.
   *
   * <p>Scanned once and cached: {@link #isEog(int)} is a native call, and a vocabulary runs to
   * six figures, so repeating the scan per token would cost more than whatever it is used for.
   * The result is shared and must not be mutated.
   */
  public int[] eogTokens() {
    int[] cached = eogTokens;
    if (cached == null) {
      synchronized (this) {
        cached = eogTokens;
        if (cached == null) {
          int n = nVocab();
          int[] found = new int[16];
          int count = 0;
          for (int id = 0; id < n; id++) {
            if (isEog(id)) {
              if (count == found.length) {
                found = java.util.Arrays.copyOf(found, count * 2);
              }
              found[count++] = id;
            }
          }
          cached = java.util.Arrays.copyOf(found, count);
          eogTokens = cached;
        }
      }
    }
    return cached;
  }

  /**
   * Returns the BOS (beginning-of-sentence) token text, or empty string if undefined.
   */
  public String bosTokenText() {
    int tokenId = llama_vocab_bos(this.segment);
    if (tokenId < 0) return "";
    var ptr = llama_vocab_get_text(this.segment, tokenId);
    return (ptr != null && ptr.address() != 0) ? ptr.getString(0) : "";
  }

  /**
   * Returns the EOS (end-of-sentence) token text, or empty string if undefined.
   */
  public String eosTokenText() {
    int tokenId = llama_vocab_eos(this.segment);
    if (tokenId < 0) return "";
    var ptr = llama_vocab_get_text(this.segment, tokenId);
    return (ptr != null && ptr.address() != 0) ? ptr.getString(0) : "";
  }

  public byte[] tokenToPiece(int tokenId) {
    int bufferSize = BUFFER_SIZE;
    try (Arena arena = Arena.ofConfined()) {
      // This loop handles the case where the initial buffer is too small for the token piece.
      // The native llama_token_to_piece function will tell us the required size.
      while (true) {
        var buffer = arena.allocate(ValueLayout.JAVA_BYTE, bufferSize);
        int pieceLength = llama_token_to_piece(
          this.segment,
          tokenId,
          buffer,
          (int) buffer.byteSize(),
          0,
          true
        );

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
