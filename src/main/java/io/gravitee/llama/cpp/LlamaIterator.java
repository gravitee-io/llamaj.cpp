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

import io.gravitee.llama.cpp.speculative.SpeculativeDecoding;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
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
      // KV prefix reuse: the first `start` prompt tokens' KV rows are already resident for this
      // sequence (start == 0 for a cold prompt). Wipe everything from `start` on so behavior is
      // self-contained regardless of whether the caller cleaned the sequence, then decode only
      // the suffix at absolute positions.
      int start = state.getReusePrefixTokens();
      if (!context.getMemory().seqRm(state.getSequenceId(), start, -1)) {
        // Recurrent/hybrid models (SSM, gated-deltanet attention) cannot rewind their state to
        // an arbitrary position — partial seq_rm is rejected natively. Fall back to a cold
        // prefill: full wipe, no reuse. Correctness over speed; callers see reuse == 0.
        context.getMemory().seqRm(state.getSequenceId(), -1, -1);
        start = 0;
        state.clearReusePrefixTokens();
      }
      // MTP keeps embeddings enabled on the target context (the head seed is the target's
      // post-norm hidden). With embeddings on, llama.cpp forces EVERY batch token to be an
      // output row ("embeddings required but some input tokens were not marked as outputs ->
      // overriding"), turning the prefill into an O(prompt) lm_head + embedding extraction
      // instead of O(1). Only the LAST prompt token's hidden is ever needed (the MTP seed),
      // so decode the bulk of the prompt with embeddings temporarily off and the final token
      // in its own single-token batch with embeddings restored — a single output row, no
      // native override.
      boolean mtpEmbeddings = state.isMtp();
      int bulkEnd = mtpEmbeddings ? totalTokens - 1 : totalTokens;
      if (mtpEmbeddings) {
        context.setEmbeddings(false);
      }
      try {
        int offset = start;
        while (offset < bulkEnd) {
          int chunkSize = Math.min(batchSize, bulkEnd - offset);
          LlamaBatch promptBatch = new LlamaBatch(arena, chunkSize, 0, 1);

          // Add tokens to the batch for the current chunk.
          for (int i = 0; i < chunkSize; i++) {
            int tokenId = state
              .getTokenized()
              .data()
              .getAtIndex(JAVA_INT, offset + i);
            // We only need the logits for the very last token of the prompt to sample the
            // next one (on the MTP path that token is decoded separately below).
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
      } finally {
        if (mtpEmbeddings) {
          // Restore for the final prompt token (seed extraction) and the verify rounds.
          context.setEmbeddings(true);
        }
      }

      if (mtpEmbeddings && totalTokens > start) {
        // Final prompt token: its logits sample the first token and its embedding row seeds
        // the MTP head. A 1-token batch with logits=true satisfies output_all — no override.
        int lastToken = state
          .getTokenized()
          .data()
          .getAtIndex(JAVA_INT, totalTokens - 1);
        LlamaBatch lastBatch = new LlamaBatch(arena, 1, 0, 1);
        lastBatch.add(
          lastToken,
          totalTokens - 1,
          java.util.List.of(state.getSequenceId()),
          true
        );
        if (lastBatch.decode(context) != 0) {
          lastBatch.free();
          throw new LlamaException(
            "Failed to decode prompt for sequence " + state.getSequenceId()
          );
        }
        lastBatch.free();
      }

      // After processing the entire prompt, update the past token count (n_past).
      state.setNPast(state.getTokenized().size());
    }

    // Sample the very first token after the prompt.
    int newToken = sampler.sample(context);
    String tokenPiece = decodeTokenPiece(state, newToken);

    // Collect logprobs if requested.
    Logprobs logprobs = collectLogprobs(state, newToken, -1);

    // Update state evaluation based on the first token (token-sequence aware: the emitted
    // text may be empty while a multi-token marker prefix is buffered).
    var emission = state
      .getStateEvaluation()
      .evaluateToken(state.getGenerationState(), newToken, tokenPiece);
    state.setGenerationState(emission.state());

    // Track the consumption of the first token (resolved tokens only; buffered marker-prefix
    // tokens are tracked when the marker is confirmed or refuted).
    state
      .getTokenTracking()
      .consume(
        new io.gravitee.llama.cpp.modules.TokenTracking.Context(
          state.getGenerationState(),
          emission.emitTokens()
        )
      );

    // Check if the generation finished immediately (e.g., if the prompt was just an EOG token).
    if (!tokenizer.isEog(newToken)) {
      // If not finished, set the new token and piece for the next iteration.
      state.setNewTokenId(newToken);
      state.setPiece(emission.emit());
      state.setPieceTokens(emission.emitTokens());
      state.setLogprobs(logprobs);
    } else {
      // If finished, set the stop reason.
      state.setFinishReason(FinishReason.STOP);
    }
  }

  /**
   * Processes a sampled token for a given state.
   * Updates state evaluation (token-sequence aware), checks for tool calls, and tracks tokens.
   *
   * @param state The conversation state to update
   * @param tokenId The sampled token id
   * @param tokenPiece The token piece that was sampled
   * @return the emission for this step: text to emit (may be empty while a multi-token marker
   *         prefix is buffered, or cover several buffered pieces on resolution) and the number
   *         of generated tokens it covers
   */
  protected io.gravitee.llama.cpp.modules.StateEvaluation.Emission processSampledToken(
    ConversationState state,
    int tokenId,
    String tokenPiece
  ) {
    // Update state evaluation
    GenerationState previousState = state.getGenerationState();
    var emission = state
      .getStateEvaluation()
      .evaluateToken(previousState, tokenId, tokenPiece);
    state.setGenerationState(emission.state());

    // Mark tool call as finished once we leave the tools section — via its close marker
    // (→ ANSWER) or a chained cross-transition directly into another state (Harmony-style).
    //
    // Only when something was actually captured in there. With a chained grammar whose markers
    // share a prefix — Harmony's reasoning-close and tool-open agree for 34 characters
    // ("<|end|><|start|>assistant<|channel|>") — the shared run buffers, the machine provisionally
    // enters TOOLS, and the text then resolves to the OTHER marker and transitions straight back
    // out. That round trip emits no tool tokens, so reporting TOOL_CALL would announce a tool call
    // the model never made. Downstream that is not a harmless label: a tool_calls finish with an
    // empty span invites callers to hunt for a call in the plain answer text and manufacture one.
    if (
      previousState == GenerationState.TOOLS &&
      emission.state() != GenerationState.TOOLS &&
      state.getTokenTracking().getOutputTokenCount(GenerationState.TOOLS) > 0
    ) {
      state.setFinishReason(FinishReason.TOOL_CALL);
    }

    // Track tokens (resolved tokens only; buffered marker-prefix tokens are tracked in the
    // state they resolve to)
    state
      .getTokenTracking()
      .consume(
        new io.gravitee.llama.cpp.modules.TokenTracking.Context(
          state.getGenerationState(),
          emission.emitTokens()
        )
      );
    return emission;
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
        // An EOG sampled while the budget ramp was boosting it is not a natural stop — the cap
        // caused it. Report LENGTH so callers still see the answer was cut short.
        state.setFinishReason(
          state.isEogRampApplied() ? FinishReason.LENGTH : FinishReason.STOP
        );
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

  /* ----- speculative decoding (see SpeculativeDecoding) ----- */

  /**
   * Brings the state's draft-side speculation state in line with the freshly-processed prompt
   * (model-draft KV replay, MTP hidden seed, or EAGLE3 prompt encode). No-op for n-gram and
   * non-speculative states. Text-only (speculative + multimodal is unsupported).
   */
  protected void prefillDraft(ConversationState state) {
    if (state.isSpeculative()) {
      state.getSpeculativeDecoding().prefill(state);
    }
  }

  /**
   * Runs one speculative draft → verify → accept round for the state's flavour (model draft,
   * n-gram, MTP, or EAGLE3) and returns the committed tokens as outputs — see
   * {@link SpeculativeDecoding}. Requires {@code state.getNewTokenId()} to be set (the last
   * token, not yet in any KV); updates the state's {@code nPast} and {@code newTokenId}, and
   * sets the finish reason on EOG or token-limit.
   */
  protected List<LlamaOutput> speculativeRound(ConversationState state) {
    return state.getSpeculativeDecoding().round(this, state);
  }

  /** Logit row for batch output {@code idx}, reinterpreted as {@code nVocab} floats. */
  /** Characters that end a sentence — always a legal place to stop. */
  private static final String SENTENCE_END = ".!?\u2026\u3002\uFF01\uFF1F";

  /** Characters that end a clause — accepted only once the ramp is past halfway. */
  private static final String CLAUSE_END = ",;:\u2014\u2013)]}\"'\u201d\u00bb";

  /**
   * Whether the text is somewhere it can stop.
   *
   * <p>The gate is the difference between a landing and an early cut. Measured: a bias applied at
   * every step makes EOG win wherever it happens to overtake the next word, which is mid-clause
   * ({@code "As the vapor travels, it"}) — the severed output the budget was supposed to avoid,
   * only sooner. Restricted to boundaries, the bias can only take an exit the text had already
   * reached.
   *
   * <p>Widening with {@code progress} keeps the early ramp conservative — full stops only — and
   * accepts clause breaks near the cap, where stopping at a comma still beats stopping mid-word.
   * At the cap everything is accepted: that is the backstop, and the hard limit would sever the
   * output at the same place anyway.
   *
   * @param lastNonSpace last non-whitespace character emitted, {@code 0} if none
   * @param endsLine     whether the last emitted piece ended a line
   * @param progress     position within the ramp, 0 at the soft threshold and 1 at the cap
   */
  static boolean atStoppingPoint(
    char lastNonSpace,
    boolean endsLine,
    float progress
  ) {
    if (progress >= 1f) {
      return true;
    }
    if (lastNonSpace == 0) {
      return false;
    }
    // A line break is a paragraph or list-item boundary: as good a stop as a full stop, and the
    // only one available in structured output (tables, bullet lists) that rarely ends sentences.
    if (endsLine || SENTENCE_END.indexOf(lastNonSpace) >= 0) {
      return true;
    }
    return progress >= 0.5f && CLAUSE_END.indexOf(lastNonSpace) >= 0;
  }

  /** Position within the ramp: 0 at or below the soft threshold, 1 at or past the cap. */
  static float eogRampProgress(int used, int maxTokens, float startFraction) {
    if (maxTokens <= 0 || startFraction < 0f) {
      return 0f;
    }
    if (used >= maxTokens) {
      return 1f;
    }
    float soft = maxTokens * startFraction;
    float span = maxTokens - soft;
    if (span <= 0f) {
      return used >= soft ? 1f : 0f;
    }
    return Math.max(0f, Math.min(1f, (used - soft) / span));
  }

  /**
   * The EOG logit boost for a given point in the budget: zero until {@code startFraction} of
   * {@code maxTokens} is spent, then quadratic to {@code maxBias} at the cap.
   *
   * <p>Quadratic rather than linear so the first tokens past the threshold are barely nudged and
   * the pressure arrives late — the model keeps its natural phrasing for most of the ramp and only
   * finds EOG overwhelming at the end. Pure arithmetic, so it is testable without a model.
   */
  static float eogRampBias(
    int used,
    int maxTokens,
    float startFraction,
    float maxBias
  ) {
    if (maxTokens <= 0 || startFraction < 0f) {
      return 0f;
    }
    // At or past the cap the bias is full, checked before the threshold test so a degenerate
    // ramp (startFraction == 1, i.e. soft == maxTokens) still lands on maxBias rather than 0.
    if (used >= maxTokens) {
      return maxBias;
    }
    float soft = maxTokens * startFraction;
    if (used <= soft) {
      return 0f;
    }
    float span = maxTokens - soft;
    if (span <= 0f) {
      return maxBias;
    }
    float t = Math.min(1f, (used - soft) / span);
    return maxBias * t * t;
  }

  /**
   * Applies the budget-aware EOG boost to the logits row this step will sample from, and records
   * on the state whether it fired. Must run immediately before sampling: {@code
   * llama_sampler_sample} reads this same native buffer, so writing here is equivalent to a bias
   * sampler at the head of the chain and composes with top-k/top-p/temperature downstream.
   */
  protected void applyEogRamp(ConversationState state, int batchPos) {
    state.setEogRampApplied(false);
    if (!state.hasEogRamp()) {
      return;
    }
    var row = logitsRow(
      state.getContext(),
      batchPos,
      state.getTokenizer().getVocab().nVocab()
    );
    if (biasEogRow(state, row, state.getAnswerTokens())) {
      state.setEogRampApplied(true);
    }
  }

  /**
   * Where the text will stand after {@code count} drafted tokens are appended.
   *
   * @param lastNonSpace last non-whitespace character at that point
   * @param endsLine     whether the text ends a line at that point
   */
  public record TextBoundary(char lastNonSpace, boolean endsLine) {}

  /**
   * Projects the stopping-point boundary forward over drafted tokens.
   *
   * <p>A speculative round biases every verify row before a single token is emitted, so the gate
   * would otherwise judge row {@code i} using text from before the round — stale by up to
   * {@code nDraft} tokens. Measured effect: the ramp never landed under speculation at all, because
   * the gate opened where the text was mid-word and stayed shut where a sentence had just ended.
   * The drafted tokens are already known here, and row {@code i} is only ever used if drafts
   * {@code 0..i-1} were accepted, so folding their text in is exact for every row that matters.
   *
   * <p>Decodes the raw piece bytes rather than the state's streaming decoder, which must not be
   * advanced out of band. A piece split mid-codepoint yields a replacement character — non-space,
   * so it reads as "not a boundary", which errs toward leaving the model's phrasing alone.
   */
  public TextBoundary projectBoundary(
    ConversationState state,
    int[] drafted,
    int count
  ) {
    char lastNonSpace = state.getLastNonSpaceChar();
    boolean endsLine = state.isLastEndsLine();
    for (int i = 0; i < count; i++) {
      byte[] bytes = state.getTokenizer().tokenToPiece(drafted[i]);
      if (bytes == null || bytes.length == 0) {
        continue;
      }
      String piece = new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
      if (piece.isEmpty()) {
        continue;
      }
      endsLine = piece.charAt(piece.length() - 1) == '\n';
      for (int j = piece.length() - 1; j >= 0; j--) {
        char c = piece.charAt(j);
        if (!Character.isWhitespace(c)) {
          lastNonSpace = c;
          break;
        }
      }
    }
    return new TextBoundary(lastNonSpace, endsLine);
  }

  /**
   * Raises every EOG logit in {@code row} for a projected budget position, returning whether any
   * bias was applied.
   *
   * <p>Split out for speculative verification, which evaluates several future positions in one
   * decode: row {@code i} of a verify batch is the distribution after {@code i} more tokens, so it
   * takes {@code used + i} rather than the current count.
   *
   * <p>Safe under speculation — and deliberately not skipped. Speculative sampling draws exactly
   * from the target distribution whatever that distribution is; the requirement is only that the
   * acceptance test and the residual draw see the SAME target. Both derive from this row, so
   * biasing it here satisfies that and the round remains exact with respect to the biased target.
   * The draft has no knowledge of the bias, so it keeps proposing continuations near the cap and
   * acceptance dips for the last few tokens — a throughput cost, not a correctness one.
   */
  public boolean biasEogRow(
    ConversationState state,
    MemorySegment row,
    int used
  ) {
    return biasEogRow(
      state,
      row,
      used,
      new TextBoundary(state.getLastNonSpaceChar(), state.isLastEndsLine())
    );
  }

  /** As above, judging the gate at an explicitly projected boundary. */
  public boolean biasEogRow(
    ConversationState state,
    MemorySegment row,
    int used,
    TextBoundary boundary
  ) {
    if (!state.hasEogRamp()) {
      return false;
    }
    float bias = eogRampBias(
      used,
      state.getMaxTokens(),
      state.getEogRampStart(),
      state.getEogRampMaxBias()
    );
    if (bias <= 0f) {
      return false;
    }
    // Only where the text can stop. Under speculation the projected rows share the current
    // boundary — the future text is unknown — so a verify row may be biased one position early
    // or late; it costs an accepted token, never correctness.
    if (
      !atStoppingPoint(
        boundary.lastNonSpace(),
        boundary.endsLine(),
        eogRampProgress(used, state.getMaxTokens(), state.getEogRampStart())
      )
    ) {
      return false;
    }
    for (int id : state.getTokenizer().getVocab().eogTokens()) {
      row.setAtIndex(
        ValueLayout.JAVA_FLOAT,
        id,
        row.getAtIndex(ValueLayout.JAVA_FLOAT, id) + bias
      );
    }
    return true;
  }

  public MemorySegment logitsRow(LlamaContext ctx, int idx, int nVocab) {
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
  public boolean emitSpeculative(
    ConversationState state,
    int token,
    List<LlamaOutput> out
  ) {
    return emitSpeculative(state, token, out, false);
  }

  /**
   * @param rowBiased whether the EOG ramp biased the logits row this token was drawn from. Per
   *                  row, not per round: a round biases several projected positions, and only the
   *                  row that actually produced the token says anything about why it appeared.
   */
  public boolean emitSpeculative(
    ConversationState state,
    int token,
    List<LlamaOutput> out,
    boolean rowBiased
  ) {
    if (state.getTokenizer().isEog(token)) {
      if (state.getFinishReason() != TOOL_CALL) {
        state.setFinishReason(rowBiased ? LENGTH : STOP);
      }
      state.setFinished(true);
      flushPendingMarker(state, out);
      return false;
    }
    String piece = decodeTokenPiece(state, token);
    // updates generation state + token tracking; may buffer or resolve marker pieces
    var emission = processSampledToken(state, token, piece);
    // Advance the stopping-point state exactly as the autoregressive path does. Without this the
    // ramp's boundary gate reads punctuation from before the round and opens at the wrong
    // positions — so whether the ramp works at all would depend on speculation being enabled.
    state.setPiece(emission.emit());
    // Mirror the autoregressive iterator: the token that reaches the quota is counted but
    // NOT emitted (the AR path stops via hasNotReachedQuota() after incrementing). Drop it
    // here too, otherwise speculative emits one extra token at the boundary.
    int max = state.getMaxTokens();
    if (max != -1 && state.getAnswerTokens() >= max) {
      state.setFinishReason(LENGTH);
      state.setFinished(true);
      return false;
    }
    // Buffered marker prefixes emit nothing, and confirmed marker text is suppressed
    // (empty emit with emitTokens > 0 — counted, never streamed).
    if (!emission.emit().isEmpty()) {
      out.add(
        new LlamaOutput(
          emission.emit(),
          emission.emitTokens(),
          state.getSequenceId()
        )
      );
    }
    return true;
  }

  /**
   * Emits (and tracks) any buffered marker-prefix pieces when generation ends while a
   * multi-token marker was still unconfirmed — the buffered text belongs to the current
   * channel and must not be dropped.
   */
  protected void flushPendingMarker(
    ConversationState state,
    List<LlamaOutput> out
  ) {
    var flush = state
      .getStateEvaluation()
      .flushPending(state.getGenerationState());
    // A flush can now close a span: when generation stops on a marker that is also the
    // prefix of a longer alternative (<|call|> ending an agent turn), the held match is
    // settled here. Adopt the emitted channel so the turn is not reported as still inside
    // the tool call, and so its tokens are attributed to the right one.
    state.setGenerationState(flush.state());
    if (flush.emitTokens() > 0) {
      state
        .getTokenTracking()
        .consume(
          new io.gravitee.llama.cpp.modules.TokenTracking.Context(
            flush.state(),
            flush.emitTokens()
          )
        );
      out.add(
        new LlamaOutput(flush.emit(), flush.emitTokens(), state.getSequenceId())
      );
    }
  }

  /**
   * Called when iteration completes.
   * Automatically cleans up the sequence from KV cache and frees the speculative state's persistent
   * native scratch (idempotent).
   */
  protected void onFinished() {
    if (currentState.isSpeculative()) {
      currentState.freeSpeculativeScratch();
    }
    if (currentState.getFinishReason() != null && !currentState.isRetainKv()) {
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
