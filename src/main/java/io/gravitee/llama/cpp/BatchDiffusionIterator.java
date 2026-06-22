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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Incremental, multi-sequence diffusion iterator — the diffusion analogue of
 * {@link BatchIterator}, shaped to plug into the same batch-engine driver.
 *
 * <p>Each call to {@link #next()} returns one {@link DiffusionToken}: a freshly finalized
 * canvas position (carrying its {@code position}, since diffusion commits positions out of
 * order), or a per-sequence final marker. Internally, {@link #hasNext()} advances the run
 * one denoising step at a time: it packs every active canvas into a single bidirectional
 * decode (each tagged with its sequence id), lets every canvas sample and transfer from its
 * own logit slice, and queues the positions that flipped from masked to committed.
 *
 * <p>Like {@link BatchIterator}, this is <b>not</b> internally synchronized: sequences are
 * added via {@link #addState(int, int[])}, removed via {@link #removeState(int)}, and the
 * whole run stopped via {@link #stop()} — all expected to be called by a single driver
 * thread (e.g. the server batch engine) that provides the locking. Removing a sequence
 * releases its KV cache so the remaining sequences keep going.
 *
 * <p>Same limitations as {@link DiffusionGenerator}: no CFG, no stochastic transfer.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class BatchDiffusionIterator
  implements Iterator<DiffusionToken>, Freeable {

  private final LlamaContext context;
  private final DiffusionParams params;
  private final int nVocab;
  private final int maskToken;
  private final boolean shiftLogits;
  private final LlamaVocab vocab;
  private final int totalSteps;

  private final LlamaSampler sampler;
  private final LlamaTokenDataArray candidates;
  private final LlamaBatch batch;

  private final Map<Integer, Canvas> active = new LinkedHashMap<>();
  private final Deque<DiffusionToken> emitQueue = new ArrayDeque<>();

  // Per-step scratch, reused to keep the step loop allocation-free.
  private final List<Canvas> batchedScratch = new ArrayList<>();
  private int[] rowBaseScratch = new int[0];

  private boolean stopped;
  private boolean freed;

  /** Per-sequence run state: the canvas, its step counter, and a cached seq-id list. */
  private static final class Canvas {

    final DiffusionCanvasState state;
    final List<Integer> seqIds; // cached to avoid a per-token List.of(...) allocation
    int stepsDone;

    Canvas(DiffusionCanvasState state) {
      this.state = state;
      this.seqIds = List.of(state.seqId());
    }
  }

  public BatchDiffusionIterator(
    Arena arena,
    LlamaContext context,
    DiffusionParams params
  ) {
    if (!context.getModel().isDiffusion()) {
      throw new LlamaException("Model is not a diffusion model");
    }
    this.context = context;
    this.params = params;
    this.nVocab = context.nVocab();
    this.maskToken = DiffusionCanvasState.resolveMaskToken(context, params);
    this.shiftLogits = params.shiftLogits() != null
      ? params.shiftLogits()
      : DiffusionCanvasState.resolveShiftLogits(context);
    this.vocab = new LlamaVocab(context.getModel());
    this.totalSteps = params.steps();

    this.sampler = DiffusionSamplers.build(arena, params);
    this.candidates = new LlamaTokenDataArray(arena, nVocab);
    // One batch big enough to pack every concurrent canvas at once.
    this.batch = new LlamaBatch(
      arena,
      context.nSeqMax() * params.maxLength(),
      0,
      context.nSeqMax()
    );
    this.batch.enableCache();

    // Bidirectional attention for the lifetime of the iterator.
    LlamaRuntime.llama_set_causal_attn(context.segment, false);
  }

  /**
   * Registers a new sequence to be denoised, analogous to
   * {@link BatchIterator#addState(ConversationState)}. The canvas joins the batch on the
   * next {@link #hasNext()} advance. Duplicate sequence ids are ignored.
   *
   * @param seqId        The sequence id to assign this canvas
   * @param promptTokens The prompt token ids
   * @return this iterator, for chaining
   */
  public BatchDiffusionIterator addState(int seqId, int[] promptTokens) {
    if (stopped || freed) {
      throw new LlamaException("Iterator is stopped");
    }
    if (active.containsKey(seqId)) {
      return this;
    }
    if (active.size() + 1 > context.nSeqMax()) {
      throw new LlamaException(
        "Adding sequence " +
          seqId +
          " exceeds nSeqMax (" +
          context.nSeqMax() +
          ")"
      );
    }
    long tokensInFlight = (long) (active.size() + 1) * params.maxLength();
    if (
      tokensInFlight > context.nUBatch() || tokensInFlight > context.nBatch()
    ) {
      throw new LlamaException(
        "Active canvases would need nUBatch/nBatch >= " +
          tokensInFlight +
          "; got nUBatch=" +
          context.nUBatch() +
          ", nBatch=" +
          context.nBatch()
      );
    }
    active.put(
      seqId,
      new Canvas(
        new DiffusionCanvasState(
          seqId,
          promptTokens,
          maskToken,
          shiftLogits,
          params
        )
      )
    );
    return this;
  }

  /**
   * Cancels a sequence mid-run, like {@link BatchIterator#removeState(int)}: drops the
   * canvas, releases its KV cache, and discards any of its still-queued tokens.
   *
   * @param seqId The sequence id to remove
   * @return {@code true} if the sequence was active
   */
  public boolean removeState(int seqId) {
    Canvas removed = active.remove(seqId);
    if (removed == null) {
      return false;
    }
    context.getMemory().seqRm(seqId, -1, -1);
    emitQueue.removeIf(token -> token.seqId() == seqId);
    return true;
  }

  /** {@code true} if any sequence is still being denoised. */
  public boolean hasActiveSequences() {
    return !active.isEmpty();
  }

  @Override
  public boolean hasNext() {
    if (stopped || freed) {
      return false;
    }
    // Advance step-by-step until something is queued, or every sequence has finished.
    while (emitQueue.isEmpty() && !active.isEmpty()) {
      step();
    }
    return !emitQueue.isEmpty();
  }

  @Override
  public DiffusionToken next() {
    if (emitQueue.isEmpty()) {
      throw new NoSuchElementException("Call hasNext() before next()");
    }
    return emitQueue.poll();
  }

  /** Runs one denoising step across all active canvases and queues finalized positions. */
  private void step() {
    // Snapshot active canvases (finalizeCanvas mutates `active` during the apply loop).
    batch.clear();
    List<Canvas> batched = batchedScratch;
    batched.clear();
    batched.addAll(active.values());
    if (rowBaseScratch.length < batched.size()) {
      rowBaseScratch = new int[batched.size()];
    }
    int[] rowBase = rowBaseScratch;
    for (int c = 0; c < batched.size(); c++) {
      Canvas canvas = batched.get(c);
      rowBase[c] = batch.nTokens();
      int[] tokens = canvas.state.tokens();
      // Only decode this canvas's live prefix (whole canvas for timestep, current block
      // for block mode), so finished/late canvases don't pay for the whole length.
      int blockNum = canvas.stepsDone / canvas.state.stepsPerBlock();
      int decodeLen = canvas.state.decodeLength(blockNum);
      for (int pos = 0; pos < decodeLen; pos++) {
        batch.add(tokens[pos], pos, canvas.seqIds, true);
      }
    }

    if (batch.decode(context) != 0) {
      throw new LlamaException("Batched diffusion decode failed");
    }
    MemorySegment logits = LlamaRuntime.llama_get_logits(context.segment);
    if (logits == null || logits.address() == 0) {
      throw new LlamaException(
        "llama_get_logits returned NULL during diffusion step"
      );
    }
    logits = logits.reinterpret(
      (long) batch.nTokens() * nVocab * ValueLayout.JAVA_FLOAT.byteSize()
    );

    for (int c = 0; c < batched.size(); c++) {
      Canvas canvas = batched.get(c);
      int blockNum = canvas.stepsDone / canvas.state.stepsPerBlock();
      int stepInBlock = canvas.stepsDone % canvas.state.stepsPerBlock();
      if (stepInBlock == 0) {
        canvas.state.beginBlock(blockNum);
      }

      canvas.state.applyStep(
        blockNum,
        stepInBlock,
        logits,
        rowBase[c],
        nVocab,
        candidates,
        sampler
      );
      canvas.stepsDone++;

      // Queue the positions committed this step (reported directly by applyStep —
      // no canvas clone / full-length diff needed).
      int[] tokens = canvas.state.tokens();
      int[] committed = canvas.state.committedPositions();
      int nc = canvas.state.committedCount();
      for (int k = 0; k < nc; k++) {
        int pos = committed[k];
        emitQueue.add(
          DiffusionToken.of(canvas.state.seqId(), pos, piece(tokens[pos]))
        );
      }

      if (
        canvas.state.done() ||
        canvas.stepsDone >= totalSteps ||
        canvas.state.answerComplete(vocab::isEog)
      ) {
        finalizeCanvas(canvas);
      }
    }
  }

  private void finalizeCanvas(Canvas canvas) {
    int seqId = canvas.state.seqId();
    active.remove(seqId);
    context.getMemory().seqRm(seqId, -1, -1);
    emitQueue.add(DiffusionToken.finalMarker(seqId));
  }

  private String piece(int tokenId) {
    return new String(vocab.tokenToPiece(tokenId), StandardCharsets.UTF_8);
  }

  /** Stops the run, releasing every remaining sequence's KV cache. */
  public void stop() {
    if (stopped) {
      return;
    }
    stopped = true;
    for (Integer seqId : active.keySet()) {
      context.getMemory().seqRm(seqId, -1, -1);
    }
    active.clear();
    emitQueue.clear();
  }

  public Stream<DiffusionToken> stream() {
    return StreamSupport.stream(
      Spliterators.spliteratorUnknownSize(this, Spliterator.ORDERED),
      false
    );
  }

  @Override
  public void free() {
    if (freed) {
      return;
    }
    stop();
    freed = true;
    batch.free();
    sampler.free();
    // Restore causal attention so the context can be reused normally.
    LlamaRuntime.llama_set_causal_attn(context.segment, true);
  }
}
