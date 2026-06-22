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

/**
 * A single token emitted by {@link BatchDiffusionIterator}.
 *
 * <p>Unlike autoregressive output, diffusion finalizes positions out of order (the most
 * confident positions anywhere in the canvas commit first), so each token carries its
 * {@code position} in the sequence. A consumer reconstructs the text by placing
 * {@code text} at {@code position} rather than appending in arrival order.
 *
 * @param seqId        The sequence id this token belongs to
 * @param position     The canvas position that was just finalized ({@code -1} for the
 *                     final end-of-sequence marker)
 * @param text         The detokenized piece for this position (empty for the final marker)
 * @param isFinal      {@code true} for the per-sequence end marker
 * @param finishReason The reason the sequence finished (non-null only when {@code isFinal})
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record DiffusionToken(
  int seqId,
  int position,
  String text,
  boolean isFinal,
  FinishReason finishReason
) {
  static DiffusionToken of(int seqId, int position, String text) {
    return new DiffusionToken(seqId, position, text, false, null);
  }

  static DiffusionToken finalMarker(int seqId) {
    return new DiffusionToken(seqId, -1, "", true, FinishReason.STOP);
  }
}
