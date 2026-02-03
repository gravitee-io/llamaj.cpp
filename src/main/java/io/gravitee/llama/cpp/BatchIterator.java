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
public final class BatchIterator extends LlamaIterator<LlamaOutput> {

  private final LlamaContext context;
  private final LlamaBatch batch;
  private final Map<Integer, ConversationState> seqIdToState;
  private final Map<Integer, Boolean> firstTokenEmitted;
  private final Map<Integer, Integer> seqIdToBatchPos;
  private final List<LlamaOutput> currentOutputs = new ArrayList<>();
  private int currentOutputIndex = 0;
  private volatile boolean stopped = false;

  /**
   * Creates a parallel batch iterator.
   * All conversation states added to this iterator MUST share the same context.
   *
   * @param arena The memory arena for allocations
   * @param context The shared context for all conversations (all states must use this same context)
   */
  public BatchIterator(Arena arena, LlamaContext context) {
    super(null); // No initial state - states are added via addState()
    this.context = context;
    this.batch = new LlamaBatch(arena, context.nBatch(), 0, context.nSeqMax());
    this.seqIdToState = new HashMap<>();
    this.firstTokenEmitted = new HashMap<>();
    this.seqIdToBatchPos = new HashMap<>();
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

    // Process the active states in smaller batches that fit the context's batch size (n_batch).
    processInBatches(activeStates);

    return !currentOutputs.isEmpty();
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

      // Clean up and remove states that have finished their generation.
      if (state.getFinishReason() != null) {
        cleanupState(state.getSequenceId());
        it.remove();
        continue;
      }

      // If the state is new, process its prompt to get the first token.
      if (state.getNewTokenId() == null) {
        processPromptForState(state);
        // If prompt processing immediately results in a finish condition, clean up.
        if (state.getFinishReason() != null) {
          cleanupState(state.getSequenceId());
          it.remove();
          continue;
        }
      }

      activeStates.add(state);

      // If the first token for this state hasn't been emitted yet, add it to the output queue.
      if (!firstTokenEmitted.get(state.getSequenceId())) {
        currentOutputs.add(
          new LlamaOutput(state.getPiece(), 1, state.getSequenceId())
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
    state.incrementNPast();
    currentOutputs.add(
      new LlamaOutput(state.getPiece(), 1, state.getSequenceId())
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
      cleanupState(state.getSequenceId());
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
      cleanupState(sequenceId);
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
    seqIdToState.values().forEach(state -> cleanupState(state.getSequenceId()));

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
      .forEach(state -> cleanupState(state.getSequenceId()));

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
   * This stops the iterator and cleans up all sequences before freeing the batch.
   */
  public void free() {
    stop();
    batch.free();
  }

  private void cleanupState(int sequenceId) {
    context.getMemory().seqRm(sequenceId, -1, -1);
    firstTokenEmitted.remove(sequenceId);
    seqIdToBatchPos.remove(sequenceId);
  }
}
