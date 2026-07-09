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

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

import io.gravitee.llama.cpp.LlamaBatch;
import io.gravitee.llama.cpp.LlamaContext;
import io.gravitee.llama.cpp.LlamaException;
import io.gravitee.llama.cpp.LlamaExt;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * MTP (nextn) self-speculation draft source — the model's own multi-token-prediction head
 * proposes draft tokens, no separate draft model (requires {@code n_layer_nextn > 0}).
 *
 * <p>Counterpart of {@link NgramIndex} for the MTP flavour: this class owns the draft-step
 * mechanics (dual token+embd decode on the MTP context, hidden-state chaining via the staging
 * nextn API), while accept/reject stays in {@code Speculation} and orchestration in the
 * iterator's MTP round. The MTP context itself (built from the <i>target</i> model with
 * {@code ctx_type=MTP}, {@code ctx_other=target}, {@code n_rs_seq>0}) is created and owned by
 * the caller, like the model-draft context.
 *
 * <p>The seed is the target's post-norm hidden state at the last committed position
 * ({@code nPast-1}); drafts beyond the first chain on the MTP context's own nextn hidden
 * ({@link LlamaExt#getEmbeddingsNextnIth}). Requires embeddings enabled on the target context
 * (done by {@link io.gravitee.llama.cpp.ConversationState#setMtp}) and the staging nextn symbols
 * ({@link LlamaExt#available()}).
 *
 * @author GraviteeSource Team
 */
public final class MtpDraft implements HiddenStateDraft {

  private final Arena arena;
  private final LlamaContext mtp;
  private final int nEmbd;

  // Persistent native scratch (lazily built, reused across rounds, freed once by free()).
  private MemorySegment embd; // single-row hidden-state buffer injected into the batch
  private LlamaBatch batch; // 1-token dual token+embd batch

  // The target's post-norm hidden at the last committed position — the seed for draft step 0.
  private float[] seed;

  public MtpDraft(Arena arena, LlamaContext mtpContext, int nEmbd) {
    this.arena = arena;
    this.mtp = mtpContext;
    this.nEmbd = nEmbd;
  }

  @Override
  public LlamaContext context() {
    return mtp;
  }

  public boolean hasSeed() {
    return seed != null;
  }

  public float[] seed() {
    return seed;
  }

  public void setSeed(float[] hidden) {
    this.seed = hidden;
  }

  /**
   * One MTP draft step: decode {@code (token @ pos)} with {@code hidden} injected as the batch's
   * embd row. The caller reads the resulting logits from batch row {@code 0} and, when chaining,
   * the next hidden via {@link #chainHidden()}.
   */
  @Override
  public void step(int token, int pos, List<Integer> seq, float[] hidden) {
    LlamaBatch b = batch();
    MemorySegment.copy(hidden, 0, embd, JAVA_FLOAT, 0, nEmbd);
    b.clear();
    b.add(token, pos, seq, true);
    if (b.decode(mtp) != 0) {
      throw new LlamaException("MTP draft decode failed");
    }
  }

  /** The MTP context's own nextn hidden for the row just decoded — seed for the next chained draft. */
  @Override
  public int hiddenSize() {
    return nEmbd;
  }

  @Override
  public float[] chainHidden(int row) {
    return LlamaExt.getEmbeddingsNextnIth(mtp, row, nEmbd);
  }

  private LlamaBatch batch() {
    if (batch == null) {
      embd = arena.allocate((long) nEmbd * Float.BYTES);
      batch = new LlamaBatch(arena, 1, 0, 1);
      // Dual token+embd batch: llama_batch_init allocated only the token array; inject our
      // persistent arena-owned hidden buffer as the embd pointer (survives clear()).
      LlamaExt.setBatchEmbd(batch, embd);
    }
    return batch;
  }

  /**
   * Frees the persistent native scratch exactly once (idempotent). The injected embd pointer is
   * arena-owned, NOT malloc'd — it must be nulled before {@code llama_batch_free}, which would
   * otherwise {@code free()} it and corrupt the heap.
   */
  public void free() {
    if (batch != null) {
      LlamaExt.setBatchEmbd(batch, MemorySegment.NULL);
      batch.free();
      batch = null;
    }
    embd = null;
    seed = null;
  }
}
