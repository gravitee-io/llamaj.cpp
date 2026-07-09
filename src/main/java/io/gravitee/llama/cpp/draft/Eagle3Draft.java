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
 * EAGLE3 draft source — a separate tiny trained head model (GGUF arch {@code eagle3}) proposes
 * draft tokens from the <i>target's</i> intermediate hidden states. Works with dense targets
 * that carry no nextn head.
 *
 * <p>Counterpart of {@link NgramIndex}/{@link MtpDraft} for the EAGLE3 flavour. Owns the protocol
 * mechanics (ported from llama.cpp {@code common/speculative.cpp} draft-eagle3, single-sequence
 * form): per-token capture of the 3 declared target layers' input hiddens → interleaved feature
 * rows → {@code llama_encode} on the head (encoder) → {@code g_embd} rows → decoder consumes
 * shifted pairs {@code (token[P+1], g[P])} at pos P via dual token+embd batches; drafts beyond
 * the first chain on the decoder's own pre-norm hidden. Accept/reject stays in
 * {@code Speculation}; orchestration in the iterator's EAGLE3 round.
 *
 * <p>The head context is created and owned by the caller (like a model-draft context). Layer
 * capture on the target and nextn capture on the head are enabled by
 * {@link io.gravitee.llama.cpp.ConversationState#setEagle3}. Deviation from the C impl: each round wipes the head's KV
 * from the boundary and re-syncs only the accepted pairs — one extra tiny decode per round for a
 * no-duplicate-cells invariant. Dense targets only.
 *
 * @author GraviteeSource Team
 */
public final class Eagle3Draft implements HiddenStateDraft {

  private final Arena arena;
  private final LlamaContext dft;
  private final int[] layerIds;
  private final int nEmbdTgt;
  private final int nEmbdDec;
  private final int nEmbdEnc;
  private final int maxRows; // nDraft + 1: verify-round row count, also the sync-batch capacity

  // Persistent native scratch (lazily built, reused across rounds, freed once by free()).
  private MemorySegment features; // [maxRows, nEmbdEnc] interleaved target features
  private MemorySegment embdRows; // [maxRows, nEmbdDec] decoder embd rows
  private LlamaBatch dualBatch; // up-to-maxRows dual token+embd decoder batch
  private LlamaBatch encShell; // raw embd-only encoder batch (all arrays NULLed)

  // Pending boundary g row: g at position nPast-1, paired with idLast at draft time.
  private float[] boundary;

  public Eagle3Draft(
    Arena arena,
    LlamaContext eagle3Context,
    int[] layerIds,
    int nEmbdTgt,
    int nEmbdDec,
    int nDraft
  ) {
    this.arena = arena;
    this.dft = eagle3Context;
    this.layerIds = layerIds;
    this.nEmbdTgt = nEmbdTgt;
    this.nEmbdDec = nEmbdDec;
    this.nEmbdEnc = layerIds.length * nEmbdTgt;
    this.maxRows = nDraft + 1;
  }

  @Override
  public LlamaContext context() {
    return dft;
  }

  public boolean hasBoundary() {
    return boundary != null;
  }

  public float[] boundary() {
    return boundary;
  }

  public void setBoundary(float[] g) {
    this.boundary = g;
  }

  /**
   * Encoder pass over the target's captured layer inputs for its last decode ({@code nRows}
   * tokens): interleave the layers into {@code [nRows, 3*nEmbdTgt]} features, run
   * {@code llama_encode} on the head (chunked by its n_ubatch), and return the g_embd rows.
   */
  public float[][] encodeCaptured(LlamaContext target, int nRows) {
    return encodeCaptured(target, 0, nRows);
  }

  /**
   * As {@link #encodeCaptured(LlamaContext, int)} but reading the capture buffer starting at
   * {@code rowOffset} — a fused multi-sequence verify decode captures all sequences' rows in one
   * buffer, and each sequence encodes only its own slice (its verify-batch base offset).
   */
  public float[][] encodeCaptured(
    LlamaContext target,
    int rowOffset,
    int nRows
  ) {
    MemorySegment feat = nRows <= maxRows
      ? featuresBuf()
      : arena.allocate((long) nRows * nEmbdEnc * Float.BYTES); // prompt prefill: one-shot
    for (int k = 0; k < layerIds.length; k++) {
      MemorySegment layer = LlamaExt.getEmbeddingsLayerInp(
        target,
        layerIds[k]
      ).reinterpret((long) (rowOffset + nRows) * nEmbdTgt * Float.BYTES);
      for (int i = 0; i < nRows; i++) {
        MemorySegment.copy(
          layer,
          (long) (rowOffset + i) * nEmbdTgt * Float.BYTES,
          feat,
          ((long) i * nEmbdEnc + (long) k * nEmbdTgt) * Float.BYTES,
          (long) nEmbdTgt * Float.BYTES
        );
      }
    }

    float[][] g = new float[nRows][];
    int nUb = dft.nUbatch();
    for (int off = 0; off < nRows; off += nUb) {
      int n = Math.min(nUb, nRows - off);
      LlamaBatch shell = encShell();
      LlamaExt.setBatchNTokens(shell, n);
      LlamaExt.setBatchEmbd(
        shell,
        feat.asSlice((long) off * nEmbdEnc * Float.BYTES)
      );
      if (shell.encode(dft) != 0) {
        throw new LlamaException(
          "EAGLE3 encoder failed (n=" + n + ", off=" + off + ")"
        );
      }
      MemorySegment gBuf = LlamaExt.getEmbeddingsNextnAll(dft).reinterpret(
        (long) n * nEmbdDec * Float.BYTES
      );
      for (int i = 0; i < n; i++) {
        float[] row = new float[nEmbdDec];
        MemorySegment.copy(
          gBuf,
          JAVA_FLOAT,
          (long) i * nEmbdDec * Float.BYTES,
          row,
          0,
          nEmbdDec
        );
        g[off + i] = row;
      }
    }
    return g;
  }

  /**
   * Decoder sync of shifted pairs {@code (toks[i] @ poss[i], gRows[i])} with no logits — keeps the
   * head's KV in lockstep with committed tokens. Chunked by the persistent batch capacity.
   */
  public void syncPairs(
    int[] toks,
    int[] poss,
    float[][] gRows,
    List<Integer> seq
  ) {
    int total = toks.length;
    int off = 0;
    while (off < total) {
      int n = Math.min(maxRows, total - off);
      LlamaBatch b = dual();
      b.clear();
      for (int i = 0; i < n; i++) {
        b.add(toks[off + i], poss[off + i], seq, false);
        MemorySegment.copy(
          gRows[off + i],
          0,
          embdRows,
          JAVA_FLOAT,
          (long) i * nEmbdDec * Float.BYTES,
          nEmbdDec
        );
      }
      if (b.decode(dft) != 0) {
        throw new LlamaException("EAGLE3 decoder sync failed");
      }
      off += n;
    }
  }

  /**
   * One decoder draft step: decode {@code (token @ pos)} with {@code hidden} as its embd row,
   * logits on. Caller reads logits from batch row {@code 0} and chains via {@link #chainHidden()}.
   */
  @Override
  public void step(int token, int pos, List<Integer> seq, float[] hidden) {
    LlamaBatch b = dual();
    b.clear();
    b.add(token, pos, seq, true);
    MemorySegment.copy(hidden, 0, embdRows, JAVA_FLOAT, 0, nEmbdDec);
    if (b.decode(dft) != 0) {
      throw new LlamaException("EAGLE3 draft decode failed");
    }
  }

  /** The decoder's own pre-norm hidden for the row just decoded — seed for the next chained draft. */
  @Override
  public int hiddenSize() {
    return nEmbdDec;
  }

  @Override
  public float[] chainHidden(int row) {
    return LlamaExt.getEmbeddingsNextnIth(dft, row, nEmbdDec);
  }

  private MemorySegment featuresBuf() {
    if (features == null) {
      features = arena.allocate((long) maxRows * nEmbdEnc * Float.BYTES);
    }
    return features;
  }

  private LlamaBatch dual() {
    if (dualBatch == null) {
      embdRows = arena.allocate((long) maxRows * nEmbdDec * Float.BYTES);
      dualBatch = new LlamaBatch(arena, maxRows, 0, 1);
      LlamaExt.setBatchEmbd(dualBatch, embdRows); // arena-owned, survives clear()
    }
    return dualBatch;
  }

  private LlamaBatch encShell() {
    if (encShell == null) {
      encShell = new LlamaBatch(arena, 1, 0, 1);
      // Raw embd-only encoder batch, mirroring the C impl's stack struct: token/pos/seq/logits
      // all NULL so llama_encode auto-fills them. The originally-allocated tiny arrays leak once.
      for (String field : new String[] {
        "token",
        "pos",
        "n_seq_id",
        "seq_id",
        "logits",
      }) {
        LlamaExt.setBatchPointer(encShell, field, MemorySegment.NULL);
      }
    }
    return encShell;
  }

  /**
   * Frees the persistent native scratch exactly once (idempotent). Injected embd pointers are
   * arena-owned, NOT malloc'd — they must be nulled before {@code llama_batch_free}.
   */
  public void free() {
    if (dualBatch != null) {
      LlamaExt.setBatchEmbd(dualBatch, MemorySegment.NULL);
      dualBatch.free();
      dualBatch = null;
    }
    if (encShell != null) {
      LlamaExt.setBatchPointer(encShell, "embd", MemorySegment.NULL);
      encShell.free();
      encShell = null;
    }
    features = null;
    embdRows = null;
    boundary = null;
  }
}
