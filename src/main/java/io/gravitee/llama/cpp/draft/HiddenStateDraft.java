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
package io.gravitee.llama.cpp.draft;

import io.gravitee.llama.cpp.LlamaContext;
import java.util.List;

/**
 * A draft source that proposes tokens by decoding {@code (token, hidden-state)} pairs and
 * chaining on its own hidden output — the shape shared by {@link MtpDraft} (nextn
 * self-speculation) and {@link Eagle3Draft} (EAGLE3 head). Lets the speculative decoder drive
 * both flavours' draft chains through one code path.
 *
 * @author GraviteeSource Team
 */
public sealed interface HiddenStateDraft permits MtpDraft, Eagle3Draft {
  /** The draft context this source decodes on (logits are read from its batch row 0). */
  LlamaContext context();

  /** The width of the hidden-state rows this source consumes and produces. */
  int hiddenSize();

  /** Decode one {@code (token @ pos)} step with {@code hidden} injected as its embd row. */
  void step(int token, int pos, List<Integer> seq, float[] hidden);

  /**
   * The source's own hidden state for output row {@code row} of the last decode — the seed for
   * that sequence's next chained draft. Fused (multi-sequence) steps read one row per sequence.
   */
  float[] chainHidden(int row);

  /** Single-row convenience: {@code chainHidden(0)}. */
  default float[] chainHidden() {
    return chainHidden(0);
  }
}
