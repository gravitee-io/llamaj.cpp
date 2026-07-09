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
package io.gravitee.llama.cpp.speculative;

import io.gravitee.llama.cpp.*;
import java.util.List;

/**
 * N-gram (prompt-lookup) speculative decoding: proposals come from the committed history — no
 * draft model, no draft forward pass, no draft KV. Verify/accept treat each proposal as a
 * point-mass draft (q = 1); only the target cache is rolled back, and committed tokens are
 * appended to the history for the next round's lookup.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class NgramSpeculativeDecoding extends SpeculativeDecoding {

  public static final NgramSpeculativeDecoding INSTANCE =
    new NgramSpeculativeDecoding();

  private NgramSpeculativeDecoding() {}

  @Override
  protected List<LlamaOutput> roundImpl(
    LlamaIterator<?> it,
    ConversationState state
  ) {
    LlamaContext target = state.getContext();
    Speculation spec = state.getSpeculation();
    int seqId = state.getSequenceId();
    int nPast = state.getNPast();
    int idLast = state.getNewTokenId();
    var seq = List.of(seqId);

    int[] drafted = state.proposeNgram(state.getNDraft());
    int m = drafted.length;

    decodeVerify(spec.verifyBatch(), target, idLast, nPast, drafted, m, seq);
    // Point-mass draft: snaps == null → q = 1 accept + point-mass residual.
    Verdict v = accept(it, spec, target, drafted, null, m);

    // Roll back ONLY the target cache (no draft cache exists).
    int newNPast = nPast + v.matched() + 1;
    target.getMemory().seqRm(seqId, newNPast, -1);

    List<LlamaOutput> out = emitCommitted(it, state, drafted, v);

    // Append the committed tokens (matched drafts + the extra) to history so the next round can
    // look them up; keeps histLen == newNPast + 1 regardless of EOG/quota early stop.
    for (int i = 0; i < v.matched(); i++) {
      state.appendHistory(drafted[i]);
    }
    state.appendHistory(v.extra());

    commit(state, v, m, newNPast);
    return out;
  }
}
