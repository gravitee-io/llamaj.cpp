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
import io.gravitee.llama.cpp.draft.Eagle3Draft;
import java.util.List;

/**
 * EAGLE3 speculative decoding: a trained head model proposes tokens from the target's captured
 * layer inputs. The head keeps a shifted-pair decoder KV ({@code (token at P+1, g at P)} at
 * pos P) and a pending boundary g row completed at draft time with the freshest sampled token;
 * each round the verify rows' features are encoded and the decoder re-synced with only the
 * accepted pairs (no stale cells). Dense targets only.
 *
 * @author GraviteeSource Team
 */
public final class Eagle3SpeculativeDecoding
  extends HiddenStateSpeculativeDecoding {

  public static final Eagle3SpeculativeDecoding INSTANCE =
    new Eagle3SpeculativeDecoding();

  private Eagle3SpeculativeDecoding() {}

  /**
   * Encodes the target's captured prompt-layer features into g_embd rows and syncs the head
   * decoder with the shifted pairs {@code (token[k+1], g[k])} at pos k; the last g row becomes
   * the pending boundary. The layer capture buffer only covers the target's <b>last</b> decode,
   * so the prompt must fit in a single target batch.
   */
  @Override
  public void prefill(ConversationState state) {
    var e3 = state.getEagle3Draft();
    var target = state.getContext();
    var tokenized = state.getTokenized();
    int n = tokenized.size();
    int seqId = state.getSequenceId();
    if (n > target.nBatch()) {
      throw new LlamaException(
        "EAGLE3 requires the prompt (" +
          n +
          " tokens) to fit in one target batch (n_batch=" +
          target.nBatch() +
          "): layer capture covers only the last target decode"
      );
    }
    e3.context().getMemory().seqRm(seqId, -1, -1);
    float[][] g = e3.encodeCaptured(target, n);
    if (n > 1) {
      int[] toks = new int[n - 1];
      int[] poss = new int[n - 1];
      float[][] rows = new float[n - 1][];
      for (int k = 0; k + 1 < n; k++) {
        toks[k] = tokenized.data().getAtIndex(JAVA_INT, k + 1);
        poss[k] = k;
        rows[k] = g[k];
      }
      e3.syncPairs(toks, poss, rows, List.of(seqId));
    }
    e3.setBoundary(g[n - 1]);
  }

  @Override
  protected List<LlamaOutput> roundImpl(
    LlamaIterator<?> it,
    ConversationState state
  ) {
    LlamaContext target = state.getContext();
    Eagle3Draft e3 = state.getEagle3Draft();
    Speculation spec = state.getSpeculation();
    int seqId = state.getSequenceId();
    int nPast = state.getNPast();
    int idLast = state.getNewTokenId();
    var seq = List.of(seqId);
    // EAGLE3 decoder convention: the pair at head pos P is (token at target pos P+1, g at P).
    // The pending boundary sits at B = nPast-1, completed now with idLast (the token at nPast).
    int boundaryPos = nPast - 1;

    e3.context().getMemory().seqRm(seqId, boundaryPos, -1);
    Drafted d = draftChain(
      it,
      e3,
      spec,
      boundaryPos,
      idLast,
      e3.boundary(),
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
    Verdict v = accept(it, state, spec, target, d.tokens(), d.snaps(), d.m());

    // Encode the verify rows' captured features → g rows (gv[r] = g at target pos nPast+r).
    // Read before any further target decode overwrites the capture buffers.
    float[][] gv = e3.encodeCaptured(target, d.m() + 1);

    int newNPast = nPast + v.matched() + 1;
    target.getMemory().seqRm(seqId, newNPast, -1);

    // Head resync: wipe from the boundary, then re-decode only the accepted shifted pairs —
    // (idLast @ B, boundary g) plus (drafted[k] @ B+1+k, gv[k]) for accepted k.
    e3.context().getMemory().seqRm(seqId, boundaryPos, -1);
    int[] toks = new int[v.matched() + 1];
    int[] poss = new int[v.matched() + 1];
    float[][] rows = new float[v.matched() + 1][];
    toks[0] = idLast;
    poss[0] = boundaryPos;
    rows[0] = e3.boundary();
    for (int k = 0; k < v.matched(); k++) {
      toks[k + 1] = d.tokens()[k];
      poss[k + 1] = boundaryPos + 1 + k;
      rows[k + 1] = gv[k];
    }
    e3.syncPairs(toks, poss, rows, seq);
    e3.setBoundary(gv[v.matched()]);

    List<LlamaOutput> out = emitCommitted(it, state, d.tokens(), v);
    commit(state, v, d.tokens(), d.m(), newNPast);
    return out;
  }
}
