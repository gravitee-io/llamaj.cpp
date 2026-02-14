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
package io.gravitee.llama.cpp.modules;

import static java.util.Objects.nonNull;

import java.util.Objects;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class PromptMemory implements Consumer<Integer, String> {

  private int maxMemorySize;
  private char[] buffer;

  private int head;
  private int tail;
  private int length;

  @Override
  public boolean isInitialized() {
    return buffer != null && maxMemorySize > 0;
  }

  @Override
  public void initialize(Integer maxMemorySize) {
    if (nonNull(maxMemorySize) && maxMemorySize > this.maxMemorySize) {
      this.maxMemorySize = maxMemorySize;
      this.buffer = new char[maxMemorySize];
      this.head = 0;
      this.tail = 0;
      this.length = 0;
    }
  }

  @Override
  public void consume(String piece) {
    if (piece == null || piece.isEmpty()) {
      return;
    }

    char[] pieceChars = piece.toCharArray();
    int pieceLength = pieceChars.length;

    if (pieceLength >= maxMemorySize) {
      System.arraycopy(
        pieceChars,
        pieceLength - maxMemorySize,
        buffer,
        0,
        maxMemorySize
      );
      head = 0;
      tail = 0;
      length = maxMemorySize;
      return;
    }

    for (char pieceChar : pieceChars) {
      buffer[tail] = pieceChar;
      tail = (tail + 1) % maxMemorySize;
      if (length < maxMemorySize) {
        length++;
      } else {
        head = (head + 1) % maxMemorySize;
      }
    }
  }

  public String getMemory() {
    if (length == 0) {
      return "";
    }

    if (head <= tail) {
      return new String(buffer, head, length);
    } else {
      char[] result = new char[length];
      int firstPartLength = maxMemorySize - head;
      System.arraycopy(buffer, head, result, 0, firstPartLength);
      System.arraycopy(buffer, 0, result, firstPartLength, tail);
      return new String(result);
    }
  }
}
