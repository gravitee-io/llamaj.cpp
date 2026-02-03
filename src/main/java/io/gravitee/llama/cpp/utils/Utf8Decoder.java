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
package io.gravitee.llama.cpp.utils;

import static java.nio.charset.StandardCharsets.*;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class Utf8Decoder {

  private final CharsetDecoder delegate;
  private byte[] carry = new byte[0];
  private ByteBuffer byteBuffer;
  private CharBuffer charBuffer;

  public Utf8Decoder() {
    this(256);
  }

  public Utf8Decoder(int initialBufferSize) {
    this.delegate = UTF_8.newDecoder()
      .onMalformedInput(CodingErrorAction.REPLACE)
      .onUnmappableCharacter(CodingErrorAction.REPLACE);
    this.byteBuffer = ByteBuffer.allocate(initialBufferSize);
    this.charBuffer = CharBuffer.allocate(initialBufferSize);
  }

  /**
   * Resets the decoder and clears any carried-over bytes.
   * This should be called before processing a new, independent stream.
   */
  public void reset() {
    delegate.reset();
    carry = new byte[0];
  }

  /**
   * Decodes a chunk of bytes into a string.
   *
   * @param bytes  The byte array containing the chunk to decode.
   * @param length The number of bytes to decode from the array.
   * @return The decoded string.
   */
  public String decode(byte[] bytes, int length) {
    if (length == 0 && carry.length == 0) {
      return "";
    }

    int combinedLength = length + carry.length;

    // Ensure our reusable buffers are large enough for the combined data.
    ensureBuffersCapacity(combinedLength);

    // Prepare buffers for writing.
    byteBuffer.clear();
    charBuffer.clear();

    // Populate the byte buffer with the carry from the previous call and the new bytes.
    if (carry.length > 0) {
      byteBuffer.put(carry);
    }
    byteBuffer.put(bytes, 0, length);
    byteBuffer.flip();

    // Decode the combined byte buffer into the char buffer.
    delegate.decode(byteBuffer, charBuffer, false);

    // If any bytes remain in the buffer, they form an incomplete multi-byte character.
    // We store them in the 'carry' array for the next call.
    updateCarry(byteBuffer);

    charBuffer.flip();
    return charBuffer.toString();
  }

  /**
   * Updates the `carry` byte array with any remaining bytes from the decoded buffer.
   * These remaining bytes represent an incomplete multi-byte UTF-8 sequence that
   * needs to be carried over to the next decoding operation.
   *
   * @param decodedBuffer The ByteBuffer that was just decoded.
   */
  private void updateCarry(ByteBuffer decodedBuffer) {
    int remaining = decodedBuffer.remaining();
    if (remaining > 0) {
      carry = new byte[remaining];
      decodedBuffer.get(carry);
    } else {
      carry = new byte[0];
    }
  }

  /**
   * Ensures that the reusable byte and char buffers have enough capacity.
   * If not, a new, larger buffer is allocated.
   *
   * @param requiredCapacity The minimum required capacity.
   */
  private void ensureBuffersCapacity(int requiredCapacity) {
    if (byteBuffer.capacity() < requiredCapacity) {
      byteBuffer = ByteBuffer.allocate(requiredCapacity);
    }
    // The char buffer can, in the worst case (ASCII), be as large as the byte buffer.
    if (charBuffer.capacity() < requiredCapacity) {
      charBuffer = CharBuffer.allocate(requiredCapacity);
    }
  }
}
