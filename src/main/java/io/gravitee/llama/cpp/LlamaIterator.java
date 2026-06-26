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

import static io.gravitee.llama.cpp.FinishReason.*;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public abstract class LlamaIterator<T> implements Iterator<T> {

  protected ConversationState currentState;
  private final MtmdContext mtmdContext;

  /**
   * Creates a new iterator with the given initial state.
   */
  public LlamaIterator(
    ConversationState initialState,
    MtmdContext mtmdContext
  ) {
    this.currentState = initialState;
    this.mtmdContext = mtmdContext;
  }

  /**
   * Creates a stream that generates tokens until a finish condition is met.
   */
  public Stream<T> stream() {
    return StreamSupport.stream(
      Spliterators.spliteratorUnknownSize(this, Spliterator.ORDERED),
      false
    );
  }

  /**
   * Checks if there are more tokens to generate.
   * This is the standard implementation used by all iterators.
   *
   * @return true if there are more tokens to generate
   */
  @Override
  public boolean hasNext() {
    boolean hasNext = batch();
    if (!hasNext) {
      onFinished();
    }
    return hasNext;
  }

  /**
   * Processes one batch step using the current state.
   * Subclasses implement the actual decoding logic.
   *
   * @return true if there are more tokens to generate, false if finished
   */
  protected abstract boolean batch();

  /**
   * Processes the initial prompt for a conversation state.
   * This method decodes the prompt tokens and samples the first token.
   * Used by both DefaultLlamaIterator and ParallelBatchIterator.
   *
   * @param state The conversation state to process
   */
  protected void processPrompt(ConversationState state) {
    var arena = state.getArena();
    var context = state.getContext();
    var sampler = state.getSampler();
    var tokenizer = state.getTokenizer();

    // The prompt may be long, so we need to process it in chunks to avoid
    // exceeding the context's batch size (n_batch).
    if (mtmdContext != null && !state.getMedia().isEmpty()) {
      // Multimodal input processing — delegate entirely to the native
      // mtmd_helper_eval_chunks which handles text tokens, image/audio encoding,
      // M-RoPE 2D/1D positions, non-causal attention, and batch splitting.
      MtmdInputChunks chunks = new MtmdInputChunks(
        mtmdContext.tokenize(
          arena,
          state.getPromptText(),
          true, // addSpecial
          true, // parseSpecial
          state.getMedia()
        )
      );

      int nPast = (int) Math.max(
        0,
        state.getContext().getMemory().posMax(state.getSequenceId()) + 1
      );

      long newNPast = mtmdContext.evalChunks(
        arena,
        context,
        chunks,
        nPast,
        state.getSequenceId(),
        context.nBatch(),
        true // logitsLast
      );

      chunks.free();

      state.setNPast((int) newNPast);
    } else {
      int totalTokens = state.getTokenized().size();
      int batchSize = Math.max(1, context.nBatch());
      int offset = 0;
      while (offset < totalTokens) {
        int chunkSize = Math.min(batchSize, totalTokens - offset);
        LlamaBatch promptBatch = new LlamaBatch(arena, chunkSize, 0, 1);

        // Add tokens to the batch for the current chunk.
        for (int i = 0; i < chunkSize; i++) {
          int tokenId = state
            .getTokenized()
            .data()
            .getAtIndex(JAVA_INT, offset + i);
          // We only need the logits for the very last token of the prompt to sample the next one.
          boolean logits = (offset + i) == totalTokens - 1;
          promptBatch.add(
            tokenId,
            offset + i,
            java.util.List.of(state.getSequenceId()),
            logits
          );
        }

        // Decode the batch of prompt tokens.
        if (promptBatch.decode(context) != 0) {
          promptBatch.free();
          throw new LlamaException(
            "Failed to decode prompt for sequence " + state.getSequenceId()
          );
        }

        promptBatch.free();
        offset += chunkSize;
      }

      // After processing the entire prompt, update the past token count (n_past).
      state.setNPast(state.getTokenized().size());
    }

    // Sample the very first token after the prompt.
    int newToken = sampler.sample(context);
    String tokenPiece = decodeTokenPiece(state, newToken);

    // Collect logprobs if requested.
    Logprobs logprobs = collectLogprobs(state, newToken, -1);

    // Update state evaluation based on the first token.
    GenerationState newState = state
      .getStateEvaluation()
      .evaluate(
        new io.gravitee.llama.cpp.modules.StateEvaluation.Context(
          state.getGenerationState(),
          tokenPiece
        )
      );
    state.setGenerationState(newState);

    // Track the consumption of the first token.
    state
      .getTokenTracking()
      .consume(
        new io.gravitee.llama.cpp.modules.TokenTracking.Context(
          state.getGenerationState(),
          1
        )
      );

    // Check if the generation finished immediately (e.g., if the prompt was just an EOG token).
    if (!tokenizer.isEog(newToken)) {
      // If not finished, set the new token and piece for the next iteration.
      state.setNewTokenId(newToken);
      state.setPiece(tokenPiece);
      state.setLogprobs(logprobs);
    } else {
      // If finished, set the stop reason.
      state.setFinishReason(FinishReason.STOP);
    }
  }

  /**
   * Processes a sampled token for a given state.
   * Updates state evaluation, checks for tool calls, and tracks tokens.
   *
   * @param state The conversation state to update
   * @param tokenPiece The token piece that was sampled
   */
  protected void processSampledToken(
    ConversationState state,
    String tokenPiece
  ) {
    // Update state evaluation
    GenerationState previousState = state.getGenerationState();
    GenerationState newState = state
      .getStateEvaluation()
      .evaluate(
        new io.gravitee.llama.cpp.modules.StateEvaluation.Context(
          previousState,
          tokenPiece
        )
      );
    state.setGenerationState(newState);

    // Mark tool call as finished once we leave the tools section
    if (
      previousState == GenerationState.TOOLS &&
      newState == GenerationState.ANSWER
    ) {
      state.setFinishReason(FinishReason.TOOL_CALL);
    }

    // Track tokens
    state
      .getTokenTracking()
      .consume(
        new io.gravitee.llama.cpp.modules.TokenTracking.Context(
          state.getGenerationState(),
          1
        )
      );
  }

  /**
   * Checks if a state should finish based on token and length limits.
   * Sets appropriate finish reason if needed.
   *
   * @param state The conversation state to check
   * @param tokenId The token ID that was sampled
   * @return true if the state should continue, false if it should finish
   */
  protected boolean shouldContinue(ConversationState state, int tokenId) {
    var tokenizer = state.getTokenizer();

    // Check for end-of-generation token
    if (tokenizer.isEog(tokenId)) {
      // Preserve TOOL_CALL — the model produced tool calls and then stopped.
      // Only set STOP if no tool calls were made.
      if (state.getFinishReason() != FinishReason.TOOL_CALL) {
        state.setFinishReason(FinishReason.STOP);
      }
      state.setFinished(true);
      return false;
    }

    // Check token limit — LENGTH always overrides, even TOOL_CALL
    int maxTokens = state.getMaxTokens();
    if (maxTokens != -1 && maxTokens <= state.getAnswerTokens()) {
      state.setFinishReason(FinishReason.LENGTH);
      state.setFinished(true);
      return false;
    }

    return true;
  }

  /**
   * Helper methods for finish reason detection.
   */
  protected boolean isEog(int tokenId) {
    boolean isEog = currentState.getTokenizer().isEog(tokenId);
    if (isEog) {
      setFinishReason(STOP);
    }
    return isEog;
  }

  protected boolean hasNotReachedQuota() {
    int maxTokens = currentState.getMaxTokens();
    boolean hasNotReachedQuota =
      maxTokens == -1 || maxTokens > currentState.getAnswerTokens();
    if (!hasNotReachedQuota) {
      setFinishReason(LENGTH);
    }
    return hasNotReachedQuota;
  }

  protected boolean endWithStopString() {
    if (!currentState.getPromptMemory().isInitialized()) {
      return false;
    }

    boolean endsWithStopString = currentState
      .getStopString()
      .evaluate(currentState.getPromptMemory().getMemory());
    if (endsWithStopString) {
      setFinishReason(STOP);
    }
    return endsWithStopString;
  }

  protected void setFinishReason(FinishReason finishReason) {
    if (currentState.getFinishReason() != null) {
      if (
        !TOOL_CALL.equals(currentState.getFinishReason()) ||
        LENGTH.equals(finishReason)
      ) {
        currentState.setFinishReason(finishReason);
      }
    } else {
      currentState.setFinishReason(finishReason);
    }
  }

  protected void feedPromptMemory(String tokenPiece) {
    if (currentState.getPromptMemory().isInitialized()) {
      currentState.getPromptMemory().consume(tokenPiece);
    }
  }

  protected String decodeTokenPiece(ConversationState state, int tokenId) {
    byte[] bytes = state.getTokenizer().tokenToPiece(tokenId);
    return state.getDecoder().decode(bytes, bytes.length);
  }

  /**
   * Collects log-probability information for the sampled token if enabled.
   *
   * @param state          The conversation state (provides context, vocab, and topLogprobs setting)
   * @param sampledTokenId The token that was sampled
   * @param batchIdx       The batch output index (use {@code -1} for the last one)
   * @return A {@link Logprobs} instance, or {@code null} if logprobs are disabled
   */
  protected Logprobs collectLogprobs(
    ConversationState state,
    int sampledTokenId,
    int batchIdx
  ) {
    int topN = state.getTopLogprobs();
    if (topN <= 0) {
      return null;
    }
    return state
      .getContext()
      .getLogprobs(
        state.getTokenizer().getVocab(),
        sampledTokenId,
        batchIdx,
        topN
      );
  }

  protected void incrementTokenCount(int tokenCount) {
    currentState
      .getTokenTracking()
      .consume(
        new io.gravitee.llama.cpp.modules.TokenTracking.Context(
          currentState.getGenerationState(),
          tokenCount
        )
      );
  }

  /* ----- speculative decoding (greedy / lossless) ----- */

  /**
   * Prefills the state's draft context with the same prompt tokens so its KV cache matches
   * the target after {@link #processPrompt}. Text-only (speculative + multimodal is
   * unsupported). No-op when the state has no draft.
   */
  protected void prefillDraft(ConversationState state) {
    if (!state.hasDraft()) {
      return;
    }
    var draft = state.getDraftContext();
    var tokenized = state.getTokenized();
    int total = tokenized.size();
    int nBatch = Math.max(1, draft.nBatch());
    int seqId = state.getSequenceId();
    // Start from a clean draft KV for this seqId: with a shared draft context, a reused seqId
    // would otherwise inherit stale cells from a previous sequence (degrades accept rate; can't
    // corrupt output since the target still verifies every token).
    draft.getMemory().seqRm(seqId, -1, -1);
    int offset = 0;
    while (offset < total) {
      int chunk = Math.min(nBatch, total - offset);
      LlamaBatch batch = new LlamaBatch(state.getArena(), chunk, 0, 1);
      try {
        for (int i = 0; i < chunk; i++) {
          int tok = tokenized.data().getAtIndex(JAVA_INT, offset + i);
          batch.add(tok, offset + i, java.util.List.of(seqId), false);
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

  /**
   * Runs one greedy draft → verify → accept cycle for a speculative state and returns the
   * accepted tokens as outputs. The draft proposes {@code nDraft} tokens; the target verifies
   * them in a single decode; the longest matching prefix (plus one correction/bonus token the
   * target picks) is committed and both KV caches are rolled back to the accepted boundary.
   * Output is identical to greedy decoding on the target.
   *
   * <p>Requires {@code state.getNewTokenId()} to be set (the last token, not yet in either KV).
   * Updates the state's {@code nPast} and {@code newTokenId}; sets the finish reason on EOG or
   * token-limit.
   */
  protected List<LlamaOutput> speculativeRound(ConversationState state) {
    if (state.isNgram()) {
      return speculativeNgramRound(state);
    }
    return state.getSpeculation().isGreedy()
      ? speculativeGreedyRound(state)
      : speculativeStochasticRound(state);
  }

  /**
   * One n-gram (prompt-lookup) speculative round: propose tokens from the committed history, verify
   * them in a single target decode, accept the longest matching prefix (greedy) or via rejection
   * sampling against a point-mass draft (stochastic), then roll back ONLY the target cache — there
   * is no draft model and no draft KV. Output is identical to plain greedy / an exact target sample.
   */
  private List<LlamaOutput> speculativeNgramRound(ConversationState state) {
    LlamaContext target = state.getContext();
    Speculation spec = state.getSpeculation();
    int seqId = state.getSequenceId();
    int kMax = state.getNDraft();
    int nPast = state.getNPast();
    int idLast = state.getNewTokenId();
    int nVocab = target.nVocab();
    boolean greedy = spec.isGreedy();
    var seq = java.util.List.of(seqId);

    int[] drafted = state.proposeNgram(kMax);
    int m = drafted.length;

    LlamaSampler chain = spec.chain();
    LlamaBatch verifyBatch = spec.verifyBatch();
    List<LlamaOutput> out = new ArrayList<>();
    try {
      verifyBatch.clear();
      verifyBatch.add(idLast, nPast, seq, true);
      for (int i = 0; i < m; i++) {
        verifyBatch.add(drafted[i], nPast + 1 + i, seq, true);
      }
      if (verifyBatch.decode(target) != 0) {
        throw new LlamaException("Speculative verify decode failed");
      }

      int matched = 0;
      int extra = -1;
      if (greedy) {
        int correction = -1;
        for (int i = 0; i < m; i++) {
          int t = chain.sample(target, i);
          if (t == drafted[i]) {
            matched++;
          } else {
            correction = t;
            break;
          }
        }
        extra = matched == m ? chain.sample(target, m) : correction;
      } else {
        for (int i = 0; i < m; i++) {
          // n-gram proposes a single certain token per position → a point-mass draft (q = 1).
          if (
            spec.acceptTarget(
              chain,
              logitsRow(target, i, nVocab),
              drafted[i],
              1.0f
            )
          ) {
            matched++;
          } else {
            extra = spec.residualTargetPointMass(drafted[i]);
            break;
          }
        }
        if (matched == m) {
          extra = spec.targetSelect(chain, logitsRow(target, m, nVocab));
        }
      }

      // Roll back ONLY the target cache (no draft cache exists).
      int newNPast = nPast + matched + 1;
      target.getMemory().seqRm(seqId, newNPast, -1);

      boolean cont = true;
      for (int i = 0; i < matched && cont; i++) {
        cont = emitSpeculative(state, drafted[i], out);
      }
      if (cont) {
        emitSpeculative(state, extra, out);
      }

      // Append the committed tokens (matched drafts + the extra) to history so the next round can
      // look them up; keeps histLen == newNPast + 1 regardless of EOG/quota early stop.
      for (int i = 0; i < matched; i++) {
        state.appendHistory(drafted[i]);
      }
      state.appendHistory(extra);

      state.setNPast(newNPast);
      state.setNewTokenId(extra);
      state.recordSpeculation(m, matched);
      return out;
    } catch (RuntimeException e) {
      spec.free(); // release persistent native scratch on failure (idempotent)
      throw e;
    }
  }

  private List<LlamaOutput> speculativeGreedyRound(ConversationState state) {
    LlamaContext target = state.getContext();
    LlamaContext draft = state.getDraftContext();
    Speculation spec = state.getSpeculation();
    int seqId = state.getSequenceId();
    int draftMax = state.getNDraft();
    int nPast = state.getNPast();
    int idLast = state.getNewTokenId();
    int nVocab = target.nVocab();
    boolean adaptive = spec.isAdaptive();
    int draftMin = spec.draftMin();
    float pMin = spec.pMin();
    var seq = java.util.List.of(seqId);

    // Persistent per-state scratch, reused across rounds and freed by spec.free() on teardown.
    LlamaSampler greedy = spec.chain(); // greedy config => greedy sampler
    LlamaBatch draftBatch = spec.draftBatch();
    LlamaBatch verifyBatch = spec.verifyBatch();
    List<LlamaOutput> out = new ArrayList<>();
    float[] probOut = new float[1];
    try {
      // Draft up to draftMax tokens (positions nPast..nPast+m-1), stopping early once the draft's
      // top-token probability drops below pMin (adaptive only) — those tokens would likely be
      // rejected anyway, so drafting them wastes draft and target compute. The drafted token is
      // always the argmax, so output stays identical to plain greedy.
      int[] drafted = new int[draftMax];
      int m = 0;
      int prev = idLast;
      for (int i = 0; i < draftMax; i++) {
        draftBatch.clear();
        draftBatch.add(prev, nPast + i, seq, true);
        if (draftBatch.decode(draft) != 0) {
          throw new LlamaException("Speculative draft decode failed");
        }
        int sampled = adaptive
          ? spec.draftGreedyConfident(
            logitsRow(draft, -1, nVocab),
            nVocab,
            probOut
          )
          : greedy.sample(draft);
        drafted[m++] = sampled;
        prev = sampled;
        if (adaptive && m >= draftMin && probOut[0] < pMin) {
          break;
        }
      }

      // Verify all drafts in one target decode.
      verifyBatch.clear();
      verifyBatch.add(idLast, nPast, seq, true);
      for (int i = 0; i < m; i++) {
        verifyBatch.add(drafted[i], nPast + 1 + i, seq, true);
      }
      if (verifyBatch.decode(target) != 0) {
        throw new LlamaException("Speculative verify decode failed");
      }

      // Accept the longest matching prefix; batch index i predicts slot drafted[i].
      int matched = 0;
      int correction = -1;
      for (int i = 0; i < m; i++) {
        int t = greedy.sample(target, i);
        if (t == drafted[i]) {
          matched++;
        } else {
          correction = t;
          break;
        }
      }
      int extra = matched == m ? greedy.sample(target, m) : correction;

      // Roll back both caches to the accepted boundary.
      int newNPast = nPast + matched + 1;
      target.getMemory().seqRm(seqId, newNPast, -1);
      draft.getMemory().seqRm(seqId, newNPast, -1);

      // Fill decode, only on full accept: advance the draft KV by one so it covers position nPast+m
      // for next round (no gap). On partial accept the seqRm above already trimmed the draft past
      // nPast+matched, so a fill would be discarded — skipping it saves a draft forward pass. The
      // sampled token is discarded.
      if (matched == m) {
        draftBatch.clear();
        draftBatch.add(prev, nPast + m, seq, true);
        if (draftBatch.decode(draft) != 0) {
          throw new LlamaException("Speculative draft decode failed");
        }
      }

      boolean cont = true;
      for (int i = 0; i < matched && cont; i++) {
        cont = emitSpeculative(state, drafted[i], out);
      }
      if (cont) {
        emitSpeculative(state, extra, out);
      }

      state.setNPast(newNPast);
      state.setNewTokenId(extra);
      state.recordSpeculation(m, matched);
      return out;
    } catch (RuntimeException e) {
      spec.free(); // release persistent native scratch on failure (idempotent)
      throw e;
    }
  }

  /**
   * Rejection-sampling speculative round (temp/top-k/top-p). The native sampler chain produces
   * the per-position distributions ({@link Speculation}); accept each drafted token with
   * probability {@code min(1, p/q)}, else take a draw from the normalized residual.
   */
  private List<LlamaOutput> speculativeStochasticRound(
    ConversationState state
  ) {
    LlamaContext target = state.getContext();
    LlamaContext draft = state.getDraftContext();
    Speculation spec = state.getSpeculation();
    int seqId = state.getSequenceId();
    int k = state.getNDraft();
    int nPast = state.getNPast();
    int idLast = state.getNewTokenId();
    int nVocab = target.nVocab();
    var seq = java.util.List.of(seqId);

    boolean adaptive = spec.isAdaptive();
    int draftMin = spec.draftMin();
    float pMin = spec.pMin();

    // Persistent per-state scratch, reused across rounds and freed by spec.free() on teardown.
    LlamaSampler chain = spec.chain();
    LlamaBatch draftBatch = spec.draftBatch();
    LlamaBatch verifyBatch = spec.verifyBatch();
    List<LlamaOutput> out = new ArrayList<>();
    try {
      // Draft up to k tokens, stopping early once the draft distribution's top probability drops
      // below pMin (adaptive only). The drafted tokens and their snapshots are unchanged regardless
      // of m, so the rejection-sampling decision per accepted token stays exact.
      int[] drafted = new int[k];
      Speculation.Snapshot[] draftSnaps = new Speculation.Snapshot[k];
      int m = 0;
      int prev = idLast;
      for (int i = 0; i < k; i++) {
        draftBatch.clear();
        draftBatch.add(prev, nPast + i, seq, true);
        if (draftBatch.decode(draft) != 0) {
          throw new LlamaException("Speculative draft decode failed");
        }
        Speculation.Snapshot s = spec.draft(
          chain,
          logitsRow(draft, -1, nVocab)
        );
        drafted[m] = s.selectedId();
        draftSnaps[m] = s;
        m++;
        prev = drafted[m - 1];
        if (adaptive && m >= draftMin && s.maxProb() < pMin) {
          break;
        }
      }

      verifyBatch.clear();
      verifyBatch.add(idLast, nPast, seq, true);
      for (int i = 0; i < m; i++) {
        verifyBatch.add(drafted[i], nPast + 1 + i, seq, true);
      }
      if (verifyBatch.decode(target) != 0) {
        throw new LlamaException("Speculative verify decode failed");
      }

      int matched = 0;
      int extra = -1;
      for (int i = 0; i < m; i++) {
        if (
          spec.acceptTarget(
            chain,
            logitsRow(target, i, nVocab),
            drafted[i],
            draftSnaps[i].selectedProbability()
          )
        ) {
          matched++;
        } else {
          extra = spec.residualTargetScatter(draftSnaps[i]);
          break;
        }
      }
      if (matched == m) {
        extra = spec.targetSelect(chain, logitsRow(target, m, nVocab));
      }

      int newNPast = nPast + matched + 1;
      target.getMemory().seqRm(seqId, newNPast, -1);
      draft.getMemory().seqRm(seqId, newNPast, -1);

      // Fill decode only on full accept (see greedy round). Token discarded.
      if (matched == m) {
        draftBatch.clear();
        draftBatch.add(prev, nPast + m, seq, true);
        if (draftBatch.decode(draft) != 0) {
          throw new LlamaException("Speculative draft decode failed");
        }
      }

      boolean cont = true;
      for (int i = 0; i < matched && cont; i++) {
        cont = emitSpeculative(state, drafted[i], out);
      }
      if (cont) {
        emitSpeculative(state, extra, out);
      }

      state.setNPast(newNPast);
      state.setNewTokenId(extra);
      state.recordSpeculation(m, matched);
      return out;
    } catch (RuntimeException e) {
      spec.free(); // release persistent native scratch on failure (idempotent)
      throw e;
    }
  }

  /** Logit row for batch output {@code idx}, reinterpreted as {@code nVocab} floats. */
  protected MemorySegment logitsRow(LlamaContext ctx, int idx, int nVocab) {
    MemorySegment ptr = LlamaRuntime.llama_get_logits_ith(ctx.segment, idx);
    if (ptr == null || ptr.address() == 0) {
      throw new LlamaException("llama_get_logits_ith returned NULL");
    }
    return ptr.reinterpret((long) nVocab * ValueLayout.JAVA_FLOAT.byteSize());
  }

  /**
   * Emits one accepted speculative token (reusing the shared detokenize / token-tracking /
   * stop logic). Returns {@code false} and marks the state finished on EOG or token-limit.
   */
  protected boolean emitSpeculative(
    ConversationState state,
    int token,
    List<LlamaOutput> out
  ) {
    if (state.getTokenizer().isEog(token)) {
      if (state.getFinishReason() != TOOL_CALL) {
        state.setFinishReason(STOP);
      }
      state.setFinished(true);
      return false;
    }
    String piece = decodeTokenPiece(state, token);
    processSampledToken(state, piece); // updates generation state + token tracking
    // Mirror the autoregressive iterator: the token that reaches the quota is counted but
    // NOT emitted (the AR path stops via hasNotReachedQuota() after incrementing). Drop it
    // here too, otherwise speculative emits one extra token at the boundary.
    int max = state.getMaxTokens();
    if (max != -1 && state.getAnswerTokens() >= max) {
      state.setFinishReason(LENGTH);
      state.setFinished(true);
      return false;
    }
    out.add(new LlamaOutput(piece, 1, state.getSequenceId()));
    return true;
  }

  /**
   * Called when iteration completes.
   * Automatically cleans up the sequence from KV cache and frees the speculative state's persistent
   * native scratch (idempotent).
   */
  protected void onFinished() {
    if (currentState.isSpeculative()) {
      currentState.getSpeculation().free();
    }
    if (currentState.getFinishReason() != null) {
      currentState
        .getContext()
        .getMemory()
        .seqRm(currentState.getSequenceId(), -1, -1);
    }
  }

  /**
   * Gets performance metrics from the current state's context and sampler.
   */
  public LlamaPerformance getPerformance() {
    var context = currentState.getContext();
    var sampler = currentState.getSampler();
    var arena = currentState.getArena();
    var contextPerf = context.getPerformance(arena);
    var samplerPerf = sampler.getPerformance(arena);
    return new LlamaPerformance(contextPerf, samplerPerf);
  }
}
