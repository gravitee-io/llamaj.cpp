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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Parallel batch iterator that processes multiple conversation states simultaneously.
 * Generates tokens for multiple conversations in a single forward pass.
 *
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
   * @param maxTokens Maximum number of tokens per batch
   * @param maxSeqIds Maximum number of sequence IDs per token
   */
  public BatchIterator(Arena arena, LlamaContext context, int maxTokens, int maxSeqIds) {
    super(null); // No initial state - states are added via addState()
    this.context = context;
    this.batch = new LlamaBatch(arena, maxTokens, 0, maxSeqIds);
    this.seqIdToState = new HashMap<>();
    this.firstTokenEmitted = new HashMap<>();
    this.seqIdToBatchPos = new HashMap<>();
  }

  /**
   * Adds a conversation state to be processed in parallel.
   * If the prompt hasn't been processed yet, it will be processed automatically.
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

    if (state.getNewTokenId() == null) {
      processPromptForState(state);
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
    // Check if iterator was stopped
    if (stopped) {
      return false;
    }

    // Clear previous outputs
    currentOutputs.clear();

    // Build batch with tokens from all active states
    batch.clear();

    // Collect active states
    List<ConversationState> activeStates = seqIdToState
      .values()
      .stream()
      .filter(state -> state.getFinishReason() == null)
      .toList();

    // Clean up finished sequences and remove from tracking
    seqIdToState
      .entrySet()
      .removeIf(entry -> {
        var state = entry.getValue();
        if (state.getFinishReason() != null) {
          context.getMemory().seqRm(state.getSequenceId(), -1, -1);
          return true;
        }
        return false;
      });

    if (activeStates.isEmpty()) {
      return false;
    }

    // First, emit any first tokens that haven't been emitted yet (from prompt processing)
    activeStates
      .stream()
      .filter(state -> !firstTokenEmitted.get(state.getSequenceId()))
      .forEach(state -> {
        // This is the first token sampled after the prompt - output it
        currentOutputs.add(new LlamaOutput(state.getPiece(), 1, state.getSequenceId()));
        firstTokenEmitted.put(state.getSequenceId(), true);
      });

    // If we just emitted first tokens, return them without doing a batch decode
    if (!currentOutputs.isEmpty()) {
      return true;
    }

    // Add one token from each active state to the batch and track positions
    for (var state : activeStates) {
      // Track the batch position before adding (this is where the logits will be)
      seqIdToBatchPos.put(state.getSequenceId(), batch.nTokens());
      batch.add(state.getNewTokenId(), state.getNPast(), List.of(state.getSequenceId()), true);
    }

    // Decode batch once for all conversations
    if (batch.decode(context) != 0) {
      // Mark all states as finished on error and clean up
      for (ConversationState state : activeStates) {
        state.setFinishReason(FinishReason.STOP);
        context.getMemory().seqRm(state.getSequenceId(), -1, -1);
      }
      seqIdToState.clear();
      return false;
    }

    // Sample from each sequence using its tracked batch position
    for (var state : activeStates) {
      var sampler = state.getSampler();
      var tokenizer = state.getTokenizer();
      // Use the actual batch position for this sequence (like i_batch[i] in batched.cpp)
      int batchPos = seqIdToBatchPos.get(state.getSequenceId());
      int newToken = sampler.sample(context, batchPos);
      String tokenPiece = tokenizer.tokenToPiece(newToken);

      // Process the sampled token (update state, check tool calls, track tokens)
      processSampledToken(state, tokenPiece);

      // Check finish conditions (EOG, length limit)
      if (!shouldContinue(state, newToken)) {
        // Don't count EOG tokens - decrement the count that was added by processSampledToken
        if (tokenizer.isEog(newToken)) {
          state
            .getTokenTracking()
            .consume(new io.gravitee.llama.cpp.modules.TokenTracking.Context(state.getGenerationState(), -1));
        }
        continue;
      }

      // Update state for next iteration
      state.setNewTokenId(newToken);
      state.setPiece(tokenPiece);
      state.incrementNPast();
      // Add to outputs with sequence ID so we know which conversation this token belongs to
      currentOutputs.add(new LlamaOutput(state.getPiece(), 1, state.getSequenceId()));
    }

    return !currentOutputs.isEmpty();
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
      // Clean up KV cache for this sequence
      context.getMemory().seqRm(sequenceId, -1, -1);

      // Clean up tracking maps
      firstTokenEmitted.remove(sequenceId);
      seqIdToBatchPos.remove(sequenceId);

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
    return seqIdToState.values().stream().anyMatch(state -> state.getFinishReason() == null);
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
    seqIdToState.values().forEach(state -> context.getMemory().seqRm(state.getSequenceId(), -1, -1));

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
      .forEach(state -> context.getMemory().seqRm(state.getSequenceId(), -1, -1));

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
      throw new java.util.NoSuchElementException("No more outputs available. Call hasNext() before next().");
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
}
