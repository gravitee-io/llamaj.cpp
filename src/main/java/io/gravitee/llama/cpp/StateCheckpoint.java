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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * The recurrent (and sliding-window) part of one sequence's state after its first
 * {@code tokens.length} tokens, serialized with {@code LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY}.
 * The bytes live off-heap in {@code arena}, which {@link #free()} closes.
 *
 * @param seqId  The sequence the state was taken from
 * @param tokens The token ids of positions {@code [0, n)}
 * @param arena  Owns {@code data}
 * @param data   The native state bytes
 * @author GraviteeSource Team
 */
record StateCheckpoint(
  int seqId,
  int[] tokens,
  Arena arena,
  MemorySegment data
) implements Freeable {
  int n() {
    return tokens.length;
  }

  boolean isPrefixOf(int[] prompt) {
    int n = n();
    return n <= prompt.length && Arrays.equals(tokens, 0, n, prompt, 0, n);
  }

  long byteSize() {
    return data == null ? 0L : data.byteSize();
  }

  @Override
  public void free() {
    if (arena != null && arena.scope().isAlive()) {
      arena.close();
    }
  }

  @Override
  public boolean isFree() {
    return arena != null && !arena.scope().isAlive();
  }
}
