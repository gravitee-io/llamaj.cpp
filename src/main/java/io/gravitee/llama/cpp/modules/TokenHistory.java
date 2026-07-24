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

import java.util.Arrays;

/**
 * Committed-token history of a conversation: the token ids whose KV rows are resident for the
 * conversation's sequence, i.e. the tokens at positions {@code [0, nPast)}.
 *
 * <p>Invariant: {@code size() == nPast} at all stable points — after prompt prefill (the full
 * tokenized prompt), after each autoregressive commit ({@code nPast} increment), and after each
 * speculative round's commit. Speculative rollbacks ({@code seqRm(seq, pos, -1)}) are mirrored
 * by {@link #truncate(int)}.
 *
 * <p>Heap-only bookkeeping — no native calls. Used by KV prefix reuse to know which prompt
 * prefix is already resident when a sequence is re-initialized with
 * {@code ConversationState.initialize(prompt, reusePrefixTokens)}.
 *
 * @author GraviteeSource Team
 */
public final class TokenHistory {

  private int[] tokens = new int[256];
  private int size = 0;

  /** Restarts the history with the given tokens (the freshly tokenized prompt). */
  public void initialize(int[] promptTokens) {
    if (tokens.length < promptTokens.length) {
      tokens = new int[Math.max(promptTokens.length, tokens.length * 2)];
    }
    System.arraycopy(promptTokens, 0, tokens, 0, promptTokens.length);
    size = promptTokens.length;
  }

  /** Appends one committed token (its KV row just became resident — nPast incremented). */
  public void append(int token) {
    if (size == tokens.length) {
      tokens = Arrays.copyOf(tokens, tokens.length * 2);
    }
    tokens[size++] = token;
  }

  /** Truncates to {@code length} tokens (mirrors a KV rollback). No-op when already shorter. */
  public void truncate(int length) {
    if (length < 0) {
      throw new IllegalArgumentException("negative history length: " + length);
    }
    if (length < size) {
      size = length;
    }
  }

  /** Number of committed tokens ({@code == nPast} at stable points). */
  public int size() {
    return size;
  }

  /** Snapshot of the committed token ids, positions {@code [0, size())}. */
  public int[] toArray() {
    return Arrays.copyOf(tokens, size);
  }
}
