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
import io.gravitee.llama.cpp.draft.HiddenStateDraft;
import java.util.List;

/**
 * Base for the flavours whose drafter decodes {@code (token, hidden-state)} pairs and chains on
 * its own hidden output ({@link MtpSpeculativeDecoding}, {@link Eagle3SpeculativeDecoding}) —
 * owns their shared draft-chain loop, including greedy/confident/snapshot sampling and the
 * adaptive {@code pMin} early-stop (identical semantics to model drafting).
 *
 * @author GraviteeSource Team
 */
public abstract sealed class HiddenStateSpeculativeDecoding
  extends SpeculativeDecoding
  permits MtpSpeculativeDecoding, Eagle3SpeculativeDecoding {

  /** Drafted tokens (m of them), their snapshots when sampling (null for greedy configs). */
  record Drafted(int[] tokens, Speculation.Snapshot[] snaps, int m) {}

  /**
   * Draft chain: step k decodes {@code (prevToken @ basePos+k)} with the current hidden,
   * samples the draft, then chains on the source's own hidden.
   */
  static Drafted draftChain(
    LlamaIterator<?> it,
    HiddenStateDraft drafter,
    Speculation spec,
    int basePos,
    int idLast,
    float[] seedHidden,
    List<Integer> seq,
    ConversationState state
  ) {
    int kMax = state.getNDraft();
    int nVocab = state.getContext().nVocab();
    boolean greedy = spec.isGreedy();
    boolean adaptive = spec.isAdaptive();
    LlamaSampler chain = spec.chain();

    int[] drafted = new int[kMax];
    Speculation.Snapshot[] snaps = greedy
      ? null
      : new Speculation.Snapshot[kMax];
    float[] probOut = new float[1];
    int m = 0;
    int prev = idLast;
    float[] h = seedHidden;
    for (int i = 0; i < kMax; i++) {
      drafter.step(prev, basePos + i, seq, h);
      float conf;
      if (greedy) {
        if (adaptive) {
          drafted[m] = spec.draftGreedyConfident(
            it.logitsRow(drafter.context(), -1, nVocab),
            nVocab,
            probOut
          );
          conf = probOut[0];
        } else {
          drafted[m] = chain.sample(drafter.context());
          conf = 1.0f;
        }
      } else {
        Speculation.Snapshot s = spec.draft(
          chain,
          it.logitsRow(drafter.context(), -1, nVocab)
        );
        snaps[m] = s;
        drafted[m] = s.selectedId();
        conf = s.maxProb();
      }
      prev = drafted[m];
      m++;
      if (adaptive && m >= spec.draftMin() && conf < spec.pMin()) {
        break;
      }
      if (m < kMax) {
        h = drafter.chainHidden(); // the source's own hidden seeds the next chained draft
      }
    }
    return new Drafted(drafted, snaps, m);
  }
}
