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
import java.util.*;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class BatchIterator
  extends LlamaIterator<LlamaOutput>
  implements AutoCloseable {

  private final LlamaContext context;
  private final LlamaBatch batch;
  private final Map<Integer, ConversationState> seqIdToState;
  private final Map<Integer, Boolean> firstTokenEmitted;
  private final Map<Integer, Integer> seqIdToBatchPos;
  private final List<LlamaOutput> currentOutputs = new ArrayList<>();
  private int currentOutputIndex = 0;
  private volatile boolean stopped = false;
  private boolean freed = false;

  /**
   * Creates a parallel batch iterator.
   * All conversation states added to this iterator MUST share the same context.
   *
   * @param arena The memory arena for allocations
   * @param context The shared context for all conversations (all states must use this same context)
   */
  public BatchIterator(
    Arena arena,
    LlamaContext context,
    MtmdContext mtmdContext
  ) {
    super(null, mtmdContext); // No initial state - states are added via addState()
    this.context = context;
    this.batch = new LlamaBatch(arena, context.nBatch(), 0, context.nSeqMax());
    this.seqIdToState = new HashMap<>();
    this.firstTokenEmitted = new HashMap<>();
    this.seqIdToBatchPos = new HashMap<>();
  }

  public BatchIterator(Arena arena, LlamaContext context) {
    this(arena, context, null);
  }

  /**
   * Adds a conversation state to be processed in parallel.
   * If the prompt hasn't been processed yet, it will be processed automatically
   * on the next batch iteration.
   * This method is thread-safe and can be called while the iterator is running.
   *
   * @param state The conversation state to add
   * @return This iterator for chaining
   * @throws LlamaException if the state uses a different context or duplicate sequence ID
   */
  public BatchIterator addState(ConversationState state) {
    // Validate that all states share the same context
    if (state.getContext() != this.context) {
      throw new LlamaException(
        "All conversation states must share the same LlamaContext. " +
          "Cannot mix states from different contexts in parallel processing."
      );
    }

    // Validate that sequence ID is not already used
    if (this.seqIdToState.containsKey(state.getSequenceId())) {
      throw new LlamaException(
        "Sequence ID " +
          state.getSequenceId() +
          " is already in use. " +
          "Each conversation state must have a unique sequence ID."
      );
    }

    this.seqIdToState.put(state.getSequenceId(), state);
    this.firstTokenEmitted.put(state.getSequenceId(), false);
    return this;
  }

  /**
   * Process the prompt for a state using the shared processPrompt() method.
   * This reuses the existing prompt processing logic from LlamaIterator.
   */
  private void processPromptForState(ConversationState state) {
    // Use the shared prompt processing method from base class
    processPrompt(state);
    // Keep the draft cache (if any) in lockstep with the target after prompt prefill.
    prefillDraft(state);
    // Seed the n-gram history once the first token is sampled (n-gram mode, not finished).
    if (state.isNgram() && state.getNewTokenId() != null) {
      state.seedNgramHistory();
    }
    // The state is now ready for parallel processing
    // (newTokenId is set, nPast is updated, first token sampled)
  }

  /**
   * Checks if there are more tokens to generate from any active conversation.
   * Returns true if there are unconsumed outputs in the current batch,
   * or if a new batch can be generated.
   */
  @Override
  public boolean hasNext() {
    // Check if explicitly stopped
    if (stopped) {
      return false;
    }

    // If there are unconsumed outputs in the current batch, return true
    if (currentOutputIndex < currentOutputs.size()) {
      return true;
    }

    // All outputs consumed, reset index and try to generate next batch
    currentOutputIndex = 0;
    return super.hasNext();
  }

  @Override
  protected boolean batch() {
    if (stopped) {
      return false;
    }

    currentOutputs.clear();

    // Prepare states: clean up finished ones, process prompts for new ones, and collect active states.
    List<ConversationState> activeStates = prepareActiveStates();

    // If we just emitted the first tokens from prompt processing, return them immediately.
    // This ensures a responsive start for each conversation.
    if (!currentOutputs.isEmpty()) {
      return true;
    }

    // If there are no more active conversations, we are done.
    if (activeStates.isEmpty()) {
      return false;
    }

    // Speculative states verify together in ONE target decode (fused); non-speculative states
    // use the normal fused single-token batch.
    List<ConversationState> speculative = new ArrayList<>();
    List<ConversationState> normal = new ArrayList<>();
    for (ConversationState state : activeStates) {
      (state.isSpeculative() ? speculative : normal).add(state);
    }
    if (!speculative.isEmpty()) {
      speculativeFusedStep(speculative);
    }
    if (!normal.isEmpty()) {
      processInBatches(normal);
    }

    return !currentOutputs.isEmpty();
  }

  /**
   * Batched speculative decoding: draft all sequences (fused per shared draft context), then verify
   * ALL of them in a single target decode (each sequence's {@code [idLast, drafts…]} packed under
   * its own seq id), then accept / roll back per sequence. Greedy or rejection-sampling per the
   * state's config.
   *
   * <p>The draft phase steps every sequence in lockstep: each draft step decodes one token per still
   * drafting sequence in a single batched draft decode, so n sequences cost {@code max(nDraft)}
   * draft forward passes instead of {@code sum(nDraft+1)} (plus one fused gap-fill decode, deferred
   * to Phase D and run only for sequences that fully accepted). Sequences sharing a draft context
   * fuse together; sequences become inactive when they hit their {@code nDraft} or stop early on low
   * confidence (adaptive). Batching never changes a sequence's logits (sequences don't cross-attend),
   * so each drafted token is identical to drafting the sequence on its own.
   */
  private void speculativeFusedStep(List<ConversationState> states) {
    int n = states.size();
    int nVocab = context.nVocab();

    // Conservative pre-check: the fused target verify must hold every sequence's (nDraft + 1) tokens
    // at once. Adaptive early stop only drafts fewer, so this upper bound stays safe.
    long verifyTokens = 0;
    for (ConversationState s : states) {
      verifyTokens += s.getNDraft() + 1L;
    }
    if (verifyTokens > context.nBatch()) {
      throw new LlamaException(
        "Fused speculative verify needs n_batch >= " +
          verifyTokens +
          " (sum of nDraft+1 across sequences); got " +
          context.nBatch()
      );
    }

    int[][] drafted = new int[n][];
    Speculation.Snapshot[][] snaps = new Speculation.Snapshot[n][];
    LlamaSampler[] chains = new LlamaSampler[n];
    int[] base = new int[n];
    int[] nDrafted = new int[n];
    // Phase A — propose drafts for all sequences. Each sequence's persistent chain (and the
    // batches/buffers it uses) is owned by its Speculation and freed by Speculation.free() in
    // cleanupState() on teardown, not per round. Model-draft sequences are decoded fused per shared
    // draft context; n-gram sequences are proposed from history (no draft model, no decode).
    for (int c = 0; c < n; c++) {
      ConversationState s = states.get(c);
      Speculation spec = s.getSpeculation();
      int k = s.getNDraft();
      chains[c] = spec.chain();
      drafted[c] = new int[k];
      // n-gram uses a per-position point-mass draft, so it needs no draft snapshots.
      if (!spec.isGreedy() && !s.isNgram()) {
        snaps[c] = new Speculation.Snapshot[k];
      }
    }
    Map<LlamaContext, List<Integer>> byDraftContext = new LinkedHashMap<>();
    for (int c = 0; c < n; c++) {
      ConversationState s = states.get(c);
      if (s.hasDraft()) {
        byDraftContext
          .computeIfAbsent(s.getDraftContext(), key -> new ArrayList<>())
          .add(c);
      } else {
        // n-gram: propose up to nDraft tokens from the committed history (no draft decode).
        int[] proposed = s.proposeNgram(s.getNDraft());
        System.arraycopy(proposed, 0, drafted[c], 0, proposed.length);
        nDrafted[c] = proposed.length;
      }
    }
    for (var entry : byDraftContext.entrySet()) {
      draftGroupFused(
        states,
        entry.getKey(),
        entry.getValue(),
        chains,
        drafted,
        snaps,
        nDrafted
      );
    }

    // Phase B — one fused target decode over all sequences' drafts.
    batch.clear();
    for (int c = 0; c < n; c++) {
      ConversationState s = states.get(c);
      base[c] = batch.nTokens();
      var seq = List.of(s.getSequenceId());
      batch.add(s.getNewTokenId(), s.getNPast(), seq, true);
      for (int i = 0; i < nDrafted[c]; i++) {
        batch.add(drafted[c][i], s.getNPast() + 1 + i, seq, true);
      }
    }
    if (batch.decode(context) != 0) {
      handleDecodeError(states);
      return;
    }

    // Phase C — accept / roll back per sequence from its slice of the fused logits. Capture each
    // sequence's pre-accept nPast and accepted count so Phase D can place the deferred fill.
    int[] oldNPast = new int[n];
    int[] matched = new int[n];
    for (int c = 0; c < n; c++) {
      oldNPast[c] = states.get(c).getNPast();
      matched[c] = acceptSequence(
        states.get(c),
        chains[c],
        base[c],
        drafted[c],
        nDrafted[c],
        snaps[c],
        nVocab
      );
    }

    // Phase D — fused gap-fill, only for full-accept (still-running) sequences. On partial accept
    // the rollback in Phase C already trimmed the over-drafted draft cells, so no fill is needed;
    // skipping it saves a draft forward pass. Grouped by shared draft context like Phase A.
    for (var entry : byDraftContext.entrySet()) {
      fillGroupFused(
        states,
        entry.getKey(),
        entry.getValue(),
        drafted,
        nDrafted,
        matched,
        oldNPast
      );
    }
  }

  /**
   * Drafts a group of sequences that share one draft context, stepping them in lockstep so each
   * draft step is a single batched decode. Writes each sequence's drafted tokens (and snapshots,
   * for the rejection-sampling path) and actual drafted count into the per-sequence arrays, indexed
   * by the sequence's global position {@code c}.
   */
  private void draftGroupFused(
    List<ConversationState> states,
    LlamaContext draftContext,
    List<Integer> group,
    LlamaSampler[] chains,
    int[][] drafted,
    Speculation.Snapshot[][] snaps,
    int[] nDrafted
  ) {
    int g = group.size();
    if (g > draftContext.nBatch()) {
      throw new LlamaException(
        "Fused speculative draft needs draft n_batch >= " +
          g +
          " (sequences sharing a draft context); got " +
          draftContext.nBatch()
      );
    }
    int nVocab = context.nVocab();
    int[] prev = new int[g];
    boolean[] active = new boolean[g];
    float[] probOut = new float[1];
    int maxK = 0;
    for (int j = 0; j < g; j++) {
      ConversationState s = states.get(group.get(j));
      prev[j] = s.getNewTokenId();
      active[j] = true;
      nDrafted[group.get(j)] = 0;
      maxK = Math.max(maxK, s.getNDraft());
    }

    // Reuse the iterator's persistent batch for the fused draft packing (cleared each step). Phase B
    // (verify) and Phase D (fill) also clear it before their own use, so no stale tokens leak between
    // phases, and there is no per-round draft-batch allocation.
    int[] row = new int[g];
    for (int step = 0; step < maxK; step++) {
      batch.clear();
      int packed = 0;
      for (int j = 0; j < g; j++) {
        if (!active[j]) {
          continue;
        }
        ConversationState s = states.get(group.get(j));
        // Logits output index == batch position (every token has logits=true).
        row[j] = batch.nTokens();
        batch.add(
          prev[j],
          s.getNPast() + step,
          List.of(s.getSequenceId()),
          true
        );
        packed++;
      }
      if (packed == 0) {
        break;
      }
      if (batch.decode(draftContext) != 0) {
        throw new LlamaException("Speculative draft decode failed");
      }
      for (int j = 0; j < g; j++) {
        if (!active[j]) {
          continue;
        }
        int c = group.get(j);
        ConversationState s = states.get(c);
        Speculation spec = s.getSpeculation();
        int sampled;
        float topProb;
        if (spec.isGreedy()) {
          if (spec.isAdaptive()) {
            sampled = spec.draftGreedyConfident(
              logitsRow(draftContext, row[j], nVocab),
              nVocab,
              probOut
            );
            topProb = probOut[0];
          } else {
            sampled = chains[c].sample(draftContext, row[j]);
            topProb = 1.0f; // unused without adaptive stop
          }
        } else {
          Speculation.Snapshot ds = spec.draft(
            chains[c],
            logitsRow(draftContext, row[j], nVocab)
          );
          sampled = ds.selectedId();
          snaps[c][nDrafted[c]] = ds;
          topProb = ds.maxProb();
        }
        drafted[c][nDrafted[c]] = sampled;
        nDrafted[c]++;
        prev[j] = sampled;
        if (nDrafted[c] >= s.getNDraft()) {
          active[j] = false;
        } else if (
          spec.isAdaptive() &&
          nDrafted[c] >= spec.draftMin() &&
          topProb < spec.pMin()
        ) {
          active[j] = false;
        }
      }
    }
  }

  /**
   * Accept the longest matching prefix for one sequence and roll back both caches. Returns the
   * accepted count {@code matched} so the caller can issue the deferred (full-accept-only) gap-fill
   * without re-running the accept test (which would advance the rejection-sampling RNG twice).
   */
  private int acceptSequence(
    ConversationState s,
    LlamaSampler chain,
    int base,
    int[] drafted,
    int m,
    Speculation.Snapshot[] snaps,
    int nVocab
  ) {
    Speculation spec = s.getSpeculation();
    int seqId = s.getSequenceId();
    boolean ngram = s.isNgram();

    int matched = 0;
    int extra = -1;
    for (int i = 0; i < m; i++) {
      if (spec.isGreedy()) {
        int t = chain.sample(context, base + i);
        if (t == drafted[i]) {
          matched++;
        } else {
          extra = t;
          break;
        }
      } else {
        // n-gram proposes a single certain token per position (point-mass q = 1); model drafting
        // uses the draft snapshot's probability.
        float qOfDrafted = ngram ? 1.0f : snaps[i].selectedProbability();
        if (
          spec.acceptTarget(
            chain,
            logitsRow(context, base + i, nVocab),
            drafted[i],
            qOfDrafted
          )
        ) {
          matched++;
        } else {
          extra = ngram
            ? spec.residualTargetPointMass(drafted[i])
            : spec.residualTargetScatter(snaps[i]);
          break;
        }
      }
    }
    if (matched == m) {
      extra = spec.isGreedy()
        ? chain.sample(context, base + m)
        : spec.targetSelect(chain, logitsRow(context, base + m, nVocab));
    }

    int newNPast = s.getNPast() + matched + 1;
    context.getMemory().seqRm(seqId, newNPast, -1);
    // Roll back the draft cache only for model drafting (n-gram has none).
    if (s.hasDraft()) {
      s.getDraftContext().getMemory().seqRm(seqId, newNPast, -1);
    }

    boolean cont = true;
    for (int i = 0; i < matched && cont; i++) {
      cont = emitSpeculative(s, drafted[i], currentOutputs);
    }
    if (cont) {
      emitSpeculative(s, extra, currentOutputs);
    }

    // Append the committed tokens to the n-gram history (keeps histLen == newNPast + 1).
    if (ngram) {
      for (int i = 0; i < matched; i++) {
        s.appendHistory(drafted[i]);
      }
      s.appendHistory(extra);
    }

    s.setNPast(newNPast);
    s.setNewTokenId(extra);
    s.recordSpeculation(m, matched);
    return matched;
  }

  /**
   * Deferred gap-fill for a draft-context group: for each sequence that fully accepted its draft
   * (and is still running), decode its last drafted token at {@code oldNPast+nDrafted} so the draft
   * KV covers that position for the next round (no full-accept gap), fused into one draft decode.
   * Partial-accept sequences were already trimmed by the Phase C rollback and need no fill.
   */
  private void fillGroupFused(
    List<ConversationState> states,
    LlamaContext draftContext,
    List<Integer> group,
    int[][] drafted,
    int[] nDrafted,
    int[] matched,
    int[] oldNPast
  ) {
    int count = 0;
    for (int c : group) {
      if (matched[c] == nDrafted[c] && !states.get(c).isFinished()) {
        count++;
      }
    }
    if (count == 0) {
      return;
    }
    // Reuse the iterator's persistent batch (clear first: it still holds Phase B's verify tokens, or
    // a previous group's fill). count <= g <= draft n_batch, and g < target n_batch == batch capacity.
    batch.clear();
    for (int c : group) {
      if (matched[c] == nDrafted[c] && !states.get(c).isFinished()) {
        ConversationState s = states.get(c);
        int m = nDrafted[c];
        batch.add(
          drafted[c][m - 1],
          oldNPast[c] + m,
          List.of(s.getSequenceId()),
          true
        );
      }
    }
    if (batch.decode(draftContext) != 0) {
      throw new LlamaException("Speculative draft decode failed");
    }
  }

  /**
   * Prepares the list of active states for the next batch processing cycle.
   * This method performs several key preparatory steps:
   * 1. Iterates through all registered conversation states.
   * 2. Cleans up finished states: Any state that has a finish reason is removed from tracking, and its resources are cleaned up.
   * 3. Processes prompts: For any new state that hasn't been processed yet, this method calls the prompt processing logic.
   * 4. Emits the first token: After a prompt is processed, the very first generated token is immediately added to the output queue.
   * 5. Collects active states: All states that are still running are collected for batch decoding.
   *
   * @return A list of {@link ConversationState}s that are ready for the next decoding batch.
   */
  private List<ConversationState> prepareActiveStates() {
    List<ConversationState> activeStates = new ArrayList<>();
    for (var it = seqIdToState.entrySet().iterator(); it.hasNext(); ) {
      var entry = it.next();
      var state = entry.getValue();

      // Clean up and remove states that have truly finished generating.
      // The `finished` flag is set by shouldContinue() when EOG or LENGTH
      // is hit — this is distinct from finishReason which may be set as a
      // marker (e.g. TOOL_CALL) while the model is still producing tokens.
      if (state.isFinished()) {
        cleanupState(state);
        it.remove();
        continue;
      }

      // If the state is new, process its prompt to get the first token.
      if (state.getNewTokenId() == null) {
        processPromptForState(state);
        // If prompt processing immediately results in a finish condition, clean up.
        if (state.getFinishReason() != null) {
          cleanupState(state);
          it.remove();
          continue;
        }
      }

      activeStates.add(state);

      // If the first token for this state hasn't been emitted yet, add it to the output queue.
      if (!firstTokenEmitted.get(state.getSequenceId())) {
        currentOutputs.add(
          new LlamaOutput(
            state.getPiece(),
            1,
            state.getSequenceId(),
            null,
            state.getLogprobs()
          )
        );
        firstTokenEmitted.put(state.getSequenceId(), true);
      }
    }
    return activeStates;
  }

  /**
   * Processes the list of active states by breaking them into smaller batches and decoding them.
   *
   * @param activeStates The list of currently active conversation states.
   */
  private void processInBatches(List<ConversationState> activeStates) {
    int batchSize = Math.max(1, context.nBatch());
    for (int start = 0; start < activeStates.size(); start += batchSize) {
      int end = Math.min(start + batchSize, activeStates.size());
      List<ConversationState> batchStates = activeStates.subList(start, end);

      // Decode one batch of states.
      if (!decodeBatch(batchStates)) {
        // If decoding fails, stop further processing.
        break;
      }
    }
  }

  /**
   * Decodes a single batch of conversation states.
   *
   * @param batchStates The list of states to decode in this batch.
   * @return {@code true} if decoding was successful, {@code false} otherwise.
   */
  private boolean decodeBatch(List<ConversationState> batchStates) {
    batch.clear();
    seqIdToBatchPos.clear();

    // Add a token from each state in the sub-batch to the main batch.
    for (ConversationState state : batchStates) {
      seqIdToBatchPos.put(state.getSequenceId(), batch.nTokens());
      batch.add(
        state.getNewTokenId(),
        state.getNPast(),
        List.of(state.getSequenceId()),
        true
      );
    }

    // Perform the main decoding step.
    if (batch.decode(context) != 0) {
      handleDecodeError(batchStates);
      return false;
    }

    // Sample a new token for each state in the batch.
    for (ConversationState state : batchStates) {
      sampleAndProcessNextToken(state);
    }
    return true;
  }

  /**
   * Samples the next token for a given state and processes it.
   *
   * @param state The conversation state to process.
   */
  private void sampleAndProcessNextToken(ConversationState state) {
    int batchPos = seqIdToBatchPos.get(state.getSequenceId());
    int newToken = state.getSampler().sample(context, batchPos);
    String tokenPiece = decodeTokenPiece(state, newToken);

    // Collect logprobs before processing (logits are invalidated after next decode).
    Logprobs logprobs = collectLogprobs(state, newToken, batchPos);

    processSampledToken(state, tokenPiece);

    // Check if the generation should continue for this state.
    if (!shouldContinue(state, newToken)) {
      if (state.getTokenizer().isEog(newToken)) {
        // Decrement token count for EOG token, as it's not part of the generated content.
        state
          .getTokenTracking()
          .consume(
            new io.gravitee.llama.cpp.modules.TokenTracking.Context(
              state.getGenerationState(),
              -1
            )
          );
      }
      return;
    }

    // Update the state with the new token and add it to the output queue.
    state.setNewTokenId(newToken);
    state.setPiece(tokenPiece);
    state.setLogprobs(logprobs);
    state.incrementNPast();
    currentOutputs.add(
      new LlamaOutput(
        state.getPiece(),
        1,
        state.getSequenceId(),
        null,
        logprobs
      )
    );
  }

  /**
   * Handles a decoding error by marking all states in the batch as finished.
   *
   * @param batchStates The list of states that were part of the failed batch.
   */
  private void handleDecodeError(List<ConversationState> batchStates) {
    for (ConversationState state : batchStates) {
      state.setFinishReason(FinishReason.STOP);
      cleanupState(state);
    }
    seqIdToState.clear();
  }

  /**
   * Removes a specific conversation state and frees its KV cache.
   * Useful when a client disconnects or you want to cancel one conversation.
   *
   * @param sequenceId The sequence ID to remove
   * @return true if the sequence was removed, false if not found
   */
  public boolean removeState(int sequenceId) {
    ConversationState state = seqIdToState.remove(sequenceId);
    if (state != null) {
      cleanupState(state);
      return true;
    }
    return false;
  }

  /**
   * Checks if there are any active conversations being processed.
   *
   * @return true if there are active conversations, false otherwise
   */
  public boolean hasActiveConversations() {
    return seqIdToState
      .values()
      .stream()
      .anyMatch(state -> state.getFinishReason() == null);
  }

  /**
   * Stops the iterator and cleans up all remaining sequences.
   * After calling this method, the iterator will no longer generate tokens.
   */
  public void stop() {
    if (stopped) {
      return;
    }

    stopped = true;

    // Clean up all remaining sequences from KV cache
    seqIdToState.values().forEach(state -> cleanupState(state));

    seqIdToState.clear();
    firstTokenEmitted.clear();
    seqIdToBatchPos.clear();
  }

  /**
   * Called when stream iteration completes naturally (all sequences finished).
   * For infinite iterators, this typically only happens when stopped explicitly.
   */
  @Override
  protected void onFinished() {
    // If we naturally finish (no active states), clean up any remaining finished sequences
    // This is mostly a safety net since batch() already handles cleanup
    seqIdToState
      .values()
      .stream()
      .filter(state -> state.getFinishReason() != null)
      .forEach(state -> cleanupState(state));

    seqIdToState.clear();
    firstTokenEmitted.clear();
    seqIdToBatchPos.clear();
  }

  /**
   * Returns the next output token.
   * Each call returns one token from one conversation.
   * Must call hasNext() before calling this method.
   *
   * @return The next LlamaOutput with sequence ID and token text
   * @throws java.util.NoSuchElementException if no more outputs are available
   */
  @Override
  public LlamaOutput next() {
    if (currentOutputIndex >= currentOutputs.size()) {
      throw new java.util.NoSuchElementException(
        "No more outputs available. Call hasNext() before next()."
      );
    }
    return currentOutputs.get(currentOutputIndex++);
  }

  /**
   * Frees resources used by this iterator.
   * This stops the iterator (cleaning up all sequences, which frees their speculative scratch)
   * before freeing the batch. Idempotent — safe to call alongside {@link #close()}.
   */
  public void free() {
    if (freed) {
      return;
    }
    freed = true;
    stop();
    batch.free();
  }

  /** AutoCloseable hook for try-with-resources; delegates to {@link #free()}. */
  @Override
  public void close() {
    free();
  }

  private void cleanupState(ConversationState state) {
    int sequenceId = state.getSequenceId();
    context.getMemory().seqRm(sequenceId, -1, -1);
    // Free the speculative state's persistent native scratch (idempotent: null-on-free, so the
    // common path of removal-then-stop never double-frees). Non-speculative states have none.
    if (state.isSpeculative()) {
      state.getSpeculation().free();
    }
    firstTokenEmitted.remove(sequenceId);
    seqIdToBatchPos.remove(sequenceId);
  }
}
