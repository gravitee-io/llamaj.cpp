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

import static java.lang.foreign.ValueLayout.JAVA_INT;

import io.gravitee.llama.cpp.*;
import java.util.List;

/**
 * Model-draft speculative decoding: a separate small model (shared vocab) proposes tokens one
 * decode at a time — with optional adaptive early-stop — and its KV cache is kept in lockstep
 * with the target's (prompt replay at prefill, rollback to the accepted boundary each round,
 * and a gap-fill decode on full accept).
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class ModelDraftSpeculativeDecoding extends SpeculativeDecoding {

  public static final ModelDraftSpeculativeDecoding INSTANCE =
    new ModelDraftSpeculativeDecoding();

  private ModelDraftSpeculativeDecoding() {}

  /**
   * Replays the prompt into the draft context so its KV matches the target after
   * {@code processPrompt}. Starts from a clean draft KV for this seqId: with a shared draft
   * context, a reused seqId would otherwise inherit stale cells from a previous sequence
   * (degrades accept rate; can't corrupt output since the target still verifies every token).
   */
  @Override
  public void prefill(ConversationState state) {
    var draft = state.getDraftContext();
    var tokenized = state.getTokenized();
    int total = tokenized.size();
    int nBatch = Math.max(1, draft.nBatch());
    int seqId = state.getSequenceId();
    state.clearPendingDraftFill(); // a fresh prompt replay invalidates any deferred fill
    draft.getMemory().seqRm(seqId, -1, -1);
    int offset = 0;
    while (offset < total) {
      int chunk = Math.min(nBatch, total - offset);
      LlamaBatch batch = new LlamaBatch(state.getArena(), chunk, 0, 1);
      try {
        for (int i = 0; i < chunk; i++) {
          int tok = tokenized.data().getAtIndex(JAVA_INT, offset + i);
          batch.add(tok, offset + i, List.of(seqId), false);
        }
        if (batch.decode(draft) != 0) {
          throw new LlamaException("Draft prefill decode failed");
        }
      } finally {
        batch.free();
      }
      offset += chunk;
    }
  }

  @Override
  protected List<LlamaOutput> roundImpl(
    LlamaIterator<?> it,
    ConversationState state
  ) {
    LlamaContext target = state.getContext();
    LlamaContext draft = state.getDraftContext();
    Speculation spec = state.getSpeculation();
    int seqId = state.getSequenceId();
    int kMax = state.getNDraft();
    int nPast = state.getNPast();
    int idLast = state.getNewTokenId();
    int nVocab = target.nVocab();
    boolean greedy = spec.isGreedy();
    boolean adaptive = spec.isAdaptive();
    var seq = List.of(seqId);

    LlamaSampler chain = spec.chain();
    LlamaBatch draftBatch = spec.draftBatch();

    // Draft up to kMax tokens (positions nPast..nPast+m-1), stopping early once the draft's
    // top-token probability drops below pMin (adaptive only) — tokens drafted past the point
    // where the draft is unsure are usually rejected anyway.
    int[] drafted = new int[kMax];
    Speculation.Snapshot[] snaps = greedy
      ? null
      : new Speculation.Snapshot[kMax];
    float[] probOut = new float[1];
    int m = 0;
    int prev = idLast;
    for (int i = 0; i < kMax; i++) {
      draftBatch.clear();
      // Piggy-back the deferred fill from the previous round's full accept (see below) onto
      // this round's first draft decode — one dispatch instead of two.
      if (i == 0 && state.hasPendingDraftFill()) {
        draftBatch.add(
          state.pendingDraftFillToken(),
          state.pendingDraftFillPos(),
          seq,
          false
        );
        state.clearPendingDraftFill();
      }
      draftBatch.add(prev, nPast + i, seq, true);
      if (draftBatch.decode(draft) != 0) {
        throw new LlamaException("Speculative draft decode failed");
      }
      float conf;
      if (greedy) {
        if (adaptive) {
          drafted[m] = spec.draftGreedyConfident(
            it.logitsRow(draft, -1, nVocab),
            nVocab,
            probOut
          );
          conf = probOut[0];
        } else {
          drafted[m] = chain.sample(draft);
          conf = 1.0f;
        }
      } else {
        Speculation.Snapshot s = spec.draft(
          chain,
          it.logitsRow(draft, -1, nVocab)
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
    }

    decodeVerify(spec.verifyBatch(), target, idLast, nPast, drafted, m, seq);
    Verdict v = accept(it, state, spec, target, drafted, snaps, m);

    // Roll back both caches to the accepted boundary.
    int newNPast = nPast + v.matched() + 1;
    target.getMemory().seqRm(seqId, newNPast, -1);
    draft.getMemory().seqRm(seqId, newNPast, -1);

    // Fill, only on full accept: the draft KV must eventually cover position nPast+m (token
    // drafted[m-1]) so there is no gap before next round's drafting. Instead of a dedicated
    // decode here, defer it into the next round's first draft batch (one dispatch instead of
    // two). On partial accept the seqRm above already trimmed the draft past nPast+matched,
    // so no fill is needed. If the conversation finishes on this round the pending fill is
    // simply never consumed.
    if (v.matched() == m) {
      state.setPendingDraftFill(prev, nPast + m);
    }

    List<LlamaOutput> out = emitCommitted(it, state, drafted, v);
    commit(state, v, drafted, m, newNPast);
    return out;
  }
}
