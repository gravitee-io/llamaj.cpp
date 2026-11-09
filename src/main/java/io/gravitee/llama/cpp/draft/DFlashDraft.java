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

import static io.gravitee.llama.cpp.LlamaRuntime.LLAMA_PROCESS_TYPE_DECODE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import io.gravitee.llama.cpp.LlamaContext;
import io.gravitee.llama.cpp.LlamaException;
import io.gravitee.llama.cpp.LlamaExt;
import io.gravitee.llama.cpp.LlamaModel;
import io.gravitee.llama.cpp.LlamaRuntime;
import io.gravitee.llama.cpp.speculative.Speculation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Block-diffusion draft model state: the drafter keeps the target's layer features in its KV
 * cache, at the target positions, and drafts a whole block in one pass by denoising
 * {@code [last token, mask, mask, …]}. The {@link Kind} of the draft decides how a decoded block
 * turns into tokens. Several sequences can share one draft context: {@link #decodeBlocks} decodes
 * their blocks together.
 *
 * @author GraviteeSource Team
 */
public final class DFlashDraft {

  /** How a decoded block turns into draft tokens. */
  public enum Kind {
    /** Draft {@code i} is read from the logits of block row {@code i + 1}. */
    DFLASH,
    /**
     * A Markov head chains the block rows in the graph, the first draft is read from the anchor
     * row, and an optional confidence head predicts each draft's acceptance.
     */
    DSPARK,
    /** A candidate selector scores the top candidates of each row given the previous draft. */
    DFLASH2,
    /**
     * A correction head scores the top candidates of each row from the state of the last
     * committed token.
     */
    LILICORR,
  }

  /** One sequence's block: the last committed token at {@code pos0}, then up to {@code nDraft} drafts. */
  public record Block(
    int seqId,
    int idLast,
    int pos0,
    int nDraft,
    MemorySegment state
  ) {}

  private static final int POSITIONS_PER_ROW = 4;

  private final Arena arena;
  private final LlamaContext dft;
  private final int[] layerIds;
  private final int nEmbdTgt;
  private final int nEmbdEnc;
  private final int nEmbdDec;
  private final int blockSize;
  private final int maskToken;
  private final Kind kind;
  private final boolean sampleFromAnchor;
  private final boolean confidenceHead;
  private final int candidates;
  private final int stateRows;

  private MemorySegment features;
  private int featureRows;
  private MemorySegment blockBatch;
  private MemorySegment states;

  public DFlashDraft(
    Arena arena,
    LlamaContext dflashContext,
    LlamaModel dflashModel,
    int[] layerIds,
    int nEmbdTgt,
    int maskToken
  ) {
    this.arena = arena;
    this.dft = dflashContext;
    this.layerIds = layerIds;
    this.nEmbdTgt = nEmbdTgt;
    this.nEmbdEnc = layerIds.length * nEmbdTgt;
    this.nEmbdDec = dflashModel.nEmbd();
    this.maskToken = maskToken;
    this.blockSize = intMeta(arena, dflashModel, "dflash.block_size", 16);
    if (blockSize < 2) {
      throw new LlamaException("invalid DFlash block size: " + blockSize);
    }
    int selectorTopK = intMeta(arena, dflashModel, "dflash.selector_top_k", 0);
    int lilicorrTopK = intMeta(
      arena,
      dflashModel,
      "lilicorr.candidate_topk",
      0
    );
    String anchor = dflashModel.metaVal(arena, "dflash.sample_from_anchor");
    String confidence = dflashModel.metaVal(
      arena,
      "dflash.has_confidence_head"
    );
    if (selectorTopK > 0) {
      this.kind = Kind.DFLASH2;
      this.candidates = selectorTopK;
    } else if (lilicorrTopK > 0) {
      this.kind = Kind.LILICORR;
      this.candidates = lilicorrTopK;
    } else if (anchor != null || confidence != null) {
      this.kind = Kind.DSPARK;
      this.candidates = 0;
    } else {
      this.kind = Kind.DFLASH;
      this.candidates = 0;
    }
    this.sampleFromAnchor = kind == Kind.DSPARK && !"false".equals(anchor);
    this.confidenceHead = kind == Kind.DSPARK && "true".equals(confidence);
    if (
      candidates > 0 && (long) candidates * (candidates + 1) > (long) nEmbdDec
    ) {
      throw new LlamaException(
        "DFlash candidate count %d does not fit the draft rows (%d)".formatted(
          candidates,
          nEmbdDec
        )
      );
    }
    this.stateRows = blockSize + 1;
    if ((readsLattice() || confidenceHead) && !LlamaExt.available()) {
      throw new LlamaException(
        "This %s draft reads the nextn output, which needs the staging nextn API:\n%s".formatted(
          kind,
          LlamaExt.resolutionReport()
        )
      );
    }
    LlamaRuntime.llama_set_causal_attn(
      dft.getMemorySegment(),
      "true".equals(dflashModel.metaVal(arena, "dflash.attention.causal"))
    );
    if (readsLattice()) {
      LlamaExt.setEmbeddingsNextn(dft, true, false);
    } else if (confidenceHead) {
      LlamaExt.setEmbeddingsNextn(dft, true, true);
    }
  }

  public LlamaContext context() {
    return dft;
  }

  public Kind kind() {
    return kind;
  }

  /** Whether drafts are proposed as certain tokens, so sampled rounds verify them as point masses. */
  public boolean pointMass() {
    return kind == Kind.LILICORR;
  }

  /** Most tokens one block drafts. */
  public int maxDraft() {
    return kind == Kind.DSPARK && sampleFromAnchor ? blockSize : blockSize - 1;
  }

  /** Tokens a block of {@code nDraft} drafts decodes. */
  public int blockTokens(int nDraft) {
    return switch (kind) {
      case LILICORR -> blockSize;
      case DSPARK -> sampleFromAnchor ? Math.max(1, nDraft) : nDraft + 1;
      case DFLASH, DFLASH2 -> nDraft + 1;
    };
  }

  /** The block of a sequence whose last committed token {@code idLast} sits at {@code pos0}. */
  public Block block(int seqId, int idLast, int pos0, int nDraft) {
    return new Block(
      seqId,
      idLast,
      pos0,
      Math.min(nDraft, maxDraft()),
      kind == Kind.LILICORR ? stateRow(seqId, pos0 - 1) : null
    );
  }

  /**
   * Writes the target's captured layer features of its last decode, rows
   * {@code [row0, row0 + nRows)}, into the draft cache at positions {@code pos0..pos0+nRows-1}.
   */
  public void inject(
    LlamaContext target,
    int row0,
    int nRows,
    int pos0,
    int seqId
  ) {
    if (nRows <= 0) {
      return;
    }
    MemorySegment feat = featuresBuf(nRows);
    for (int k = 0; k < layerIds.length; k++) {
      MemorySegment layer = LlamaExt.getEmbeddingsLayerInp(
        target,
        layerIds[k]
      ).reinterpret((long) (row0 + nRows) * nEmbdTgt * Float.BYTES);
      for (int i = 0; i < nRows; i++) {
        MemorySegment.copy(
          layer,
          (long) (row0 + i) * nEmbdTgt * Float.BYTES,
          feat,
          ((long) i * nEmbdEnc + (long) k * nEmbdTgt) * Float.BYTES,
          (long) nEmbdTgt * Float.BYTES
        );
      }
    }
    int nUb = dft.nUbatch();
    for (int off = 0; off < nRows; off += nUb) {
      int n = Math.min(nUb, nRows - off);
      MemorySegment batch = LlamaRuntime.llama_batch_ext_init(
        dft.getMemorySegment()
      );
      try (var local = Arena.ofConfined()) {
        MemorySegment pos = local.allocate(JAVA_INT, POSITIONS_PER_ROW);
        for (int i = 0; i < n; i++) {
          MemorySegment embd = LlamaRuntime.llama_embd_allocate(local);
          LlamaRuntime.llama_embd_set(
            embd,
            feat.asSlice((long) (off + i) * nEmbdEnc * Float.BYTES),
            1,
            nEmbdEnc
          );
          int idx = LlamaRuntime.llama_batch_ext_add_embd(batch, seqId, embd);
          if (idx < 0) {
            throw new LlamaException(
              "DFlash feature batch rejected a row: %d".formatted(idx)
            );
          }
          int p = pos0 + off + i;
          pos.setAtIndex(JAVA_INT, 0, p);
          pos.setAtIndex(JAVA_INT, 1, p);
          pos.setAtIndex(JAVA_INT, 2, p);
          pos.setAtIndex(JAVA_INT, 3, 0);
          if (
            !LlamaRuntime.llama_batch_ext_set_pos(batch, idx, pos) ||
            !LlamaRuntime.llama_batch_ext_set_output_embd(batch, idx, false)
          ) {
            throw new LlamaException("failed to fill the DFlash feature batch");
          }
        }
        int ret = LlamaRuntime.llama_process(
          dft.getMemorySegment(),
          LLAMA_PROCESS_TYPE_DECODE,
          batch
        );
        if (ret != 0) {
          throw new LlamaException(
            "DFlash feature injection failed: %d".formatted(ret)
          );
        }
        if (kind == Kind.LILICORR) {
          keepStates(seqId, pos0 + off, n);
        }
      } finally {
        LlamaRuntime.llama_batch_ext_free(batch);
      }
    }
  }

  /**
   * Decodes the blocks of one or more sequences in one pass and returns the batch index of each
   * block's first token, in order. Remove each block with {@link #trim} once its drafts are read.
   */
  public int[] decodeBlocks(List<Block> blocks) {
    MemorySegment b = blockBatch();
    LlamaRuntime.llama_batch_ext_clear(b);
    boolean output = !readsLattice();
    int[] begs = new int[blocks.size()];
    int total = 0;
    try (var local = Arena.ofConfined()) {
      MemorySegment pos = local.allocate(JAVA_INT, POSITIONS_PER_ROW);
      MemorySegment embd = LlamaRuntime.llama_embd_allocate(local);
      for (int j = 0; j < blocks.size(); j++) {
        Block blk = blocks.get(j);
        int n = blockTokens(blk.nDraft());
        total += n;
        if (blk.state() != null) {
          LlamaRuntime.llama_embd_set(embd, blk.state(), 1, nEmbdDec);
        }
        for (int i = 0; i < n; i++) {
          int idx = LlamaRuntime.llama_batch_ext_add_token(
            b,
            blk.seqId(),
            i == 0 ? blk.idLast() : maskToken
          );
          if (idx < 0) {
            throw new LlamaException(
              "DFlash block batch rejected a token: %d".formatted(idx)
            );
          }
          if (i == 0) {
            begs[j] = idx;
          }
          pos.setAtIndex(JAVA_INT, 0, blk.pos0() + i);
          if (
            !LlamaRuntime.llama_batch_ext_set_pos(b, idx, pos) ||
            !LlamaRuntime.llama_batch_ext_set_output_logits(b, idx, output) ||
            (blk.state() != null &&
              !LlamaRuntime.llama_batch_ext_set_embd_state(b, idx, embd))
          ) {
            throw new LlamaException("failed to fill the DFlash block batch");
          }
        }
      }
      if (total > dft.nUbatch()) {
        throw new LlamaException(
          "DFlash blocks need %d tokens in one ubatch; the draft context has n_ubatch=%d".formatted(
            total,
            dft.nUbatch()
          )
        );
      }
      int ret = LlamaRuntime.llama_process(
        dft.getMemorySegment(),
        LLAMA_PROCESS_TYPE_DECODE,
        b
      );
      if (ret != 0) {
        throw new LlamaException(
          "DFlash block decode failed: %d".formatted(ret)
        );
      }
    }
    return begs;
  }

  /**
   * Reads up to {@code nDraft} drafts of the block whose first token is at batch index
   * {@code beg} into {@code drafted}, and their draft distributions into {@code snaps} for
   * sampled rounds ({@code snaps} is null for greedy rounds and {@link #pointMass()} drafts).
   * Adaptive configs stop at the first draft below {@code pMin}.
   *
   * @return the number of drafts read
   */
  public int readDrafts(
    Speculation spec,
    int beg,
    int nDraft,
    int[] drafted,
    Speculation.Snapshot[] snaps
  ) {
    return switch (kind) {
      case DFLASH -> readLogits(spec, beg + 1, nDraft, drafted, snaps, false);
      case DSPARK -> readLogits(
        spec,
        sampleFromAnchor ? beg : beg + 1,
        nDraft,
        drafted,
        snaps,
        confidenceHead
      );
      case DFLASH2, LILICORR -> readLattice(spec, beg, nDraft, drafted, snaps);
    };
  }

  /** Removes the draft cache cells of {@code seqId} at positions {@code >= from}. */
  public void trim(int seqId, int from) {
    dft.getMemory().seqRm(seqId, from, -1);
  }

  public void free() {
    if (blockBatch != null) {
      LlamaRuntime.llama_batch_ext_free(blockBatch);
      blockBatch = null;
    }
    features = null;
    states = null;
  }

  private boolean readsLattice() {
    return kind == Kind.DFLASH2 || kind == Kind.LILICORR;
  }

  private int readLogits(
    Speculation spec,
    int row0,
    int nDraft,
    int[] drafted,
    Speculation.Snapshot[] snaps,
    boolean confidence
  ) {
    int nVocab = dft.nVocab();
    boolean adaptive = spec.isAdaptive();
    MemorySegment conf = confidence && adaptive
      ? LlamaExt.getEmbeddingsNextnAll(dft).reinterpret(
        (long) (row0 + nDraft) * nEmbdDec * Float.BYTES
      )
      : null;
    float[] probOut = new float[1];
    int m = 0;
    for (int i = 0; i < nDraft; i++) {
      int row = row0 + i;
      if (
        conf != null &&
        m >= spec.draftMin() &&
        conf.getAtIndex(JAVA_FLOAT, (long) row * nEmbdDec) < spec.pMin()
      ) {
        break;
      }
      float p;
      if (spec.isGreedy()) {
        if (adaptive && conf == null) {
          drafted[m] = spec.draftGreedyConfident(
            logitsRow(row, nVocab),
            nVocab,
            probOut
          );
          p = probOut[0];
        } else {
          drafted[m] = spec.chain().sample(dft, row);
          p = 1.0f;
        }
      } else {
        Speculation.Snapshot s = spec.draft(
          spec.chain(),
          logitsRow(row, nVocab)
        );
        snaps[m] = s;
        drafted[m] = s.selectedId();
        p = s.maxProb();
      }
      m++;
      if (conf == null && adaptive && m >= spec.draftMin() && p < spec.pMin()) {
        break;
      }
    }
    return m;
  }

  private int readLattice(
    Speculation spec,
    int beg,
    int nDraft,
    int[] drafted,
    Speculation.Snapshot[] snaps
  ) {
    int k = candidates;
    int last = beg + nDraft;
    MemorySegment lattice = LlamaExt.getEmbeddingsNextnAll(dft).reinterpret(
      (long) (last + 1) * nEmbdDec * Float.BYTES
    );
    boolean sample = kind == Kind.DFLASH2 && !spec.isGreedy();
    boolean adaptive = spec.isAdaptive();
    int[] ids = new int[k];
    float[] scores = new float[k];
    int predecessor = 0;
    int m = 0;
    for (int row = beg + 1; row <= last; row++) {
      long base = (long) row * nEmbdDec;
      long from = base + k + (long) predecessor * k;
      int best = 0;
      for (int c = 0; c < k; c++) {
        ids[c] = (int) lattice.getAtIndex(JAVA_FLOAT, base + c);
        scores[c] = lattice.getAtIndex(JAVA_FLOAT, from + c);
        if (scores[c] > scores[best]) {
          best = c;
        }
      }
      if (sample) {
        Speculation.Snapshot s = spec.draftCandidates(ids, scores);
        if (adaptive && m >= spec.draftMin() && s.maxProb() < spec.pMin()) {
          break;
        }
        snaps[m] = s;
        drafted[m++] = s.selectedId();
        predecessor = indexOf(ids, s.selectedId());
      } else {
        if (
          adaptive &&
          m >= spec.draftMin() &&
          topProbability(scores, best) < spec.pMin()
        ) {
          break;
        }
        drafted[m++] = ids[best];
        predecessor = best;
      }
    }
    return m;
  }

  private static float topProbability(float[] scores, int best) {
    double sum = 0.0;
    for (float s : scores) {
      sum += Math.exp(s - scores[best]);
    }
    return (float) (1.0 / sum);
  }

  private static int indexOf(int[] ids, int id) {
    for (int i = 0; i < ids.length; i++) {
      if (ids[i] == id) {
        return i;
      }
    }
    throw new LlamaException("drafted token is not a candidate: " + id);
  }

  private MemorySegment logitsRow(int idx, int nVocab) {
    MemorySegment ptr = LlamaRuntime.llama_get_logits_ith(
      dft.getMemorySegment(),
      idx
    );
    if (ptr == null || ptr.address() == 0) {
      throw new LlamaException("llama_get_logits_ith returned NULL");
    }
    return ptr.reinterpret((long) nVocab * Float.BYTES);
  }

  private void keepStates(int seqId, int pos0, int n) {
    MemorySegment projected = LlamaExt.getEmbeddingsNextnAll(dft).reinterpret(
      (long) n * nEmbdDec * Float.BYTES
    );
    int posMin = Math.max(0, pos0 + n - 1 - blockSize);
    for (int i = 0; i < n; i++) {
      int pos = pos0 + i;
      if (pos < posMin) {
        continue;
      }
      int slot = stateSlot(seqId, pos);
      MemorySegment.copy(
        projected,
        (long) i * nEmbdDec * Float.BYTES,
        states(),
        (long) slot * nEmbdDec * Float.BYTES,
        (long) nEmbdDec * Float.BYTES
      );
    }
  }

  private MemorySegment stateRow(int seqId, int pos) {
    int slot = stateSlot(seqId, Math.max(0, pos));
    return states().asSlice(
      (long) slot * nEmbdDec * Float.BYTES,
      (long) nEmbdDec * Float.BYTES
    );
  }

  private int stateSlot(int seqId, int pos) {
    if (seqId < 0 || seqId >= dft.nSeqMax()) {
      throw new LlamaException(
        "sequence %d is outside the DFlash context (n_seq_max=%d)".formatted(
          seqId,
          dft.nSeqMax()
        )
      );
    }
    return seqId * stateRows + (pos % stateRows);
  }

  private MemorySegment states() {
    if (states == null) {
      int slots = dft.nSeqMax() * stateRows;
      states = arena.allocate((long) slots * nEmbdDec * Float.BYTES);
    }
    return states;
  }

  private MemorySegment featuresBuf(int nRows) {
    if (features == null || featureRows < nRows) {
      features = arena.allocate((long) nRows * nEmbdEnc * Float.BYTES);
      featureRows = nRows;
    }
    return features;
  }

  private MemorySegment blockBatch() {
    if (blockBatch == null) {
      blockBatch = LlamaRuntime.llama_batch_ext_init(dft.getMemorySegment());
      if (blockBatch == null || blockBatch.equals(MemorySegment.NULL)) {
        throw new LlamaException("failed to create the DFlash block batch");
      }
    }
    return blockBatch;
  }

  private static int intMeta(
    Arena arena,
    LlamaModel model,
    String key,
    int fallback
  ) {
    String value = model.metaVal(arena, key);
    if (value == null || value.isBlank()) {
      return fallback;
    }
    try {
      return Integer.parseInt(value.trim());
    } catch (NumberFormatException e) {
      throw new LlamaException("invalid %s: %s".formatted(key, value));
    }
  }
}
