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
import io.gravitee.llama.cpp.draft.DFlashDraft;
import java.util.List;

/**
 * Block-diffusion speculative decoding (DFlash, DSpark, DFlash2, LiliCorr): a draft model that
 * holds the target's layer features in its KV cache drafts a whole block in one forward pass, by
 * denoising {@code [last token, mask, …]}. Each round the accepted rows' features are written to
 * the draft cache and the rest of the block is discarded. Dense targets only.
 *
 * @author GraviteeSource Team
 */
public final class DFlashSpeculativeDecoding extends SpeculativeDecoding {

  public static final DFlashSpeculativeDecoding INSTANCE =
    new DFlashSpeculativeDecoding();

  private DFlashSpeculativeDecoding() {}

  /**
   * Writes the features of the prompt tokens the target just decoded into the draft cache; a
   * reused prefix keeps the features of the previous turn. The layer capture covers only the
   * target's last decode, so the decoded part of the prompt must fit in one target batch.
   */
  @Override
  public void prefill(ConversationState state) {
    DFlashDraft draft = state.getDFlashDraft();
    LlamaContext target = state.getContext();
    int n = state.getTokenized().size();
    int start = state.getReusedPrefixTokens();
    int decoded = n - start;
    int seqId = state.getSequenceId();
    if (decoded > target.nBatch()) {
      throw new LlamaException(
        "DFlash requires the decoded prompt (" +
          decoded +
          " tokens) to fit in one target batch (n_batch=" +
          target.nBatch() +
          "): layer capture covers only the last target decode"
      );
    }
    draft.trim(seqId, start);
    draft.inject(target, 0, decoded, start, seqId);
  }

  @Override
  protected List<LlamaOutput> roundImpl(
    LlamaIterator<?> it,
    ConversationState state
  ) {
    LlamaContext target = state.getContext();
    DFlashDraft draft = state.getDFlashDraft();
    Speculation spec = state.getSpeculation();
    int seqId = state.getSequenceId();
    int nPast = state.getNPast();
    int idLast = state.getNewTokenId();

    DFlashDraft.Block block = draft.block(
      seqId,
      idLast,
      nPast,
      state.getNDraft()
    );
    int k = block.nDraft();
    int beg = draft.decodeBlocks(List.of(block))[0];
    int[] drafted = new int[k];
    Speculation.Snapshot[] snaps = spec.isGreedy() || draft.pointMass()
      ? null
      : new Speculation.Snapshot[k];
    int m = draft.readDrafts(spec, beg, k, drafted, snaps);
    draft.trim(seqId, nPast);

    decodeVerify(
      spec.verifyBatch(),
      target,
      idLast,
      nPast,
      drafted,
      m,
      List.of(seqId)
    );
    Verdict v = accept(it, state, spec, target, drafted, snaps, m);

    int newNPast = nPast + v.matched() + 1;
    draft.inject(target, 0, v.matched() + 1, nPast, seqId);
    target.getMemory().seqRm(seqId, newNPast, -1);
    draft.trim(seqId, newNPast);

    List<LlamaOutput> out = emitCommitted(it, state, drafted, v);
    commit(state, v, drafted, m, newNPast);
    return out;
  }
}
