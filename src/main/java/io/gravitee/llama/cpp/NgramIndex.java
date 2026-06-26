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

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Committed-token history for n-gram (prompt-lookup) drafting, with an incrementally-maintained
 * position index so a proposal is ~O(1) amortized instead of an O(history) backward scan (the latter
 * makes a whole generation O(n²)).
 *
 * <p>The index maps a rolling hash of each {@code ngram}-token window to the start positions where it
 * occurs (most recent last). Appending a token indexes the one window it completes in O(ngram); a
 * proposal hashes the current window, takes the most recent earlier start with matching tokens
 * (collisions are resolved by an actual token comparison), and returns what followed it. The result
 * is identical to a straightforward backward scan (the {@code NgramLookupTest} oracle) — a wrong
 * proposal could only lower the accept rate, never change emitted tokens.
 *
 * <p>Single-threaded per conversation state, like the rest of the speculative scratch.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
final class NgramIndex {

  private static final int[] NO_DRAFT = new int[0];
  private static final long HASH_PRIME = 1099511628211L;

  private final int ngram;
  private int[] history;
  private int histLen;
  private final Map<Long, Positions> index = new HashMap<>();

  NgramIndex(int ngram) {
    this.ngram = ngram;
    this.history = new int[16];
  }

  /** Resets to empty, reusing the buffer — for a re-initialized conversation. */
  void clear() {
    histLen = 0;
    index.clear();
  }

  /** Appends one committed token and indexes the {@code ngram} window it completes (if any). */
  void append(int token) {
    if (histLen == history.length) {
      history = Arrays.copyOf(history, history.length * 2);
    }
    int p = histLen;
    history[histLen++] = token;
    if (p >= ngram - 1) {
      int start = p - ngram + 1;
      index.computeIfAbsent(hashAt(start), k -> new Positions()).add(start);
    }
  }

  /**
   * Proposes up to {@code kMax} tokens that followed the most recent earlier occurrence of the last
   * {@code ngram} tokens; empty array if there is no earlier match.
   */
  int[] propose(int kMax) {
    if (histLen < ngram + 1) {
      return NO_DRAFT;
    }
    int patStart = histLen - ngram;
    Positions ps = index.get(hashAt(patStart));
    if (ps != null) {
      for (int i = ps.size - 1; i >= 0; i--) {
        int start = ps.data[i];
        // Skip the current window itself and resolve any hash collision by comparing tokens.
        if (start >= patStart || !matches(start, patStart)) {
          continue;
        }
        int contStart = start + ngram;
        int k = Math.min(kMax, histLen - contStart);
        int[] out = new int[k];
        System.arraycopy(history, contStart, out, 0, k);
        return out;
      }
    }
    return NO_DRAFT;
  }

  private long hashAt(int start) {
    long h = 0;
    for (int t = 0; t < ngram; t++) {
      h = h * HASH_PRIME + history[start + t];
    }
    return h;
  }

  private boolean matches(int a, int b) {
    for (int t = 0; t < ngram; t++) {
      if (history[a + t] != history[b + t]) {
        return false;
      }
    }
    return true;
  }

  /** Minimal growable int list (amortized O(1) append, no boxing). */
  private static final class Positions {

    private int[] data = new int[4];
    private int size = 0;

    void add(int p) {
      if (size == data.length) {
        data = Arrays.copyOf(data, data.length * 2);
      }
      data[size++] = p;
    }
  }
}
