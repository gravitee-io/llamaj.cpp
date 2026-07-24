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
import io.gravitee.llama.cpp.draft.MtpDraft;
import java.util.List;

/**
 * MTP (nextn) self-speculative decoding: the target model's own multi-token-prediction head
 * proposes tokens — no separate draft model. The head is seeded with the target's post-norm
 * hidden at the last committed position and chains on its own nextn hidden; the seed advances
 * to the last accepted verify row each round.
 *
 * @author GraviteeSource Team
 */
public final class MtpSpeculativeDecoding
  extends HiddenStateSpeculativeDecoding {

  public static final MtpSpeculativeDecoding INSTANCE =
    new MtpSpeculativeDecoding();

  private MtpSpeculativeDecoding() {}

  /**
   * No token replay — the head shares the target's weights and is seeded per round with the
   * target's post-norm hidden. After {@code processPrompt} the target's single output row is
   * the last prompt token's hidden: the initial seed. Read via index {@code -1} (last output
   * row): a positive index maps through {@code output_ids[token_index]}, which only equals the
   * output row when every batch token requested logits (the verify path) — after a prompt
   * prefill (full or KV-prefix-reused) only the final prompt token has an output row.
   */
  @Override
  public void prefill(ConversationState state) {
    var mtp = state.getMtpDraft();
    mtp.context().getMemory().seqRm(state.getSequenceId(), -1, -1);
    mtp.setSeed(state.getContext().getEmbeddingsIth(-1));
  }

  @Override
  protected List<LlamaOutput> roundImpl(
    LlamaIterator<?> it,
    ConversationState state
  ) {
    LlamaContext target = state.getContext();
    MtpDraft mtp = state.getMtpDraft();
    Speculation spec = state.getSpeculation();
    int seqId = state.getSequenceId();
    int nPast = state.getNPast();
    int idLast = state.getNewTokenId();
    var seq = List.of(seqId);

    // Wipe stale head cells in this round's write window (previous round's chain overrun).
    mtp.context().getMemory().seqRm(seqId, nPast, -1);
    Drafted d = draftChain(
      it,
      mtp,
      spec,
      nPast,
      idLast,
      mtp.seed(),
      seq,
      state
    );

    decodeVerify(
      spec.verifyBatch(),
      target,
      idLast,
      nPast,
      d.tokens(),
      d.m(),
      seq
    );
    Verdict v = accept(it, spec, target, d.tokens(), d.snaps(), d.m());

    // Next round's seed: the target's hidden at the last ACCEPTED verify row (input position
    // nPast+matched — the position preceding the new idLast). Read before any further decode.
    float[] newSeed = target.getEmbeddingsIth(v.matched());

    int newNPast = nPast + v.matched() + 1;
    target.getMemory().seqRm(seqId, newNPast, -1);
    mtp.context().getMemory().seqRm(seqId, newNPast, -1);
    mtp.setSeed(newSeed);

    List<LlamaOutput> out = emitCommitted(it, state, d.tokens(), v);
    commit(state, v, d.tokens(), d.m(), newNPast);
    return out;
  }
}
