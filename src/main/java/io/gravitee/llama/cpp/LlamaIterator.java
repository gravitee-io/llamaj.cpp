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

import java.util.Iterator;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * A stateless iterator that processes tokens for a conversation.
 * The iterator is just a cursor that can be switched between different conversation states.
 *
 * <p>Key features:
 * <ul>
 *   <li>Owns no resources - just processes the current state</li>
 *   <li>Can be switched between conversations via {@link #setState(ConversationState)}</li>
 *   <li>Supports continuous generation for one state</li>
 *   <li>Supports round-robin processing across multiple states</li>
 * </ul>
 *
 * <p>Example - Continuous generation:
 * <pre>{@code
 * var state = ConversationState.create(context, tokenizer, sampler, 0)
 *     .setMaxTokens(100)
 *     .initialize("Hello!");
 *
 * var iterator = new LlamaIterator(state);
 * iterator.stream().forEach(output -> System.out.print(output.text()));
 * }</pre>
 *
 * <p>Example - Switching between conversations:
 * <pre>{@code
 * var conv1 = ConversationState.create(context, tokenizer, sampler, 1)
 *     .initialize("First conversation");
 * var conv2 = ConversationState.create(context, tokenizer, sampler, 2)
 *     .initialize("Second conversation");
 *
 * var iterator = new LlamaIterator(conv1);
 *
 * // Generate 5 tokens from conv1
 * for (int i = 0; i < 5 && iterator.hasNext(); i++) {
 *     System.out.print(iterator.next().text());
 * }
 *
 * // Switch to conv2
 * iterator.setState(conv2);
 * while (iterator.hasNext()) {
 *     System.out.print(iterator.next().text());
 * }
 * }</pre>
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public abstract class LlamaIterator<T> implements Iterator<T> {

  protected ConversationState currentState;

  /**
   * Creates a new iterator with the given initial state.
   */
  public LlamaIterator(ConversationState initialState) {
    this.currentState = initialState;
  }

  /**
   * Switches this iterator to process a different conversation state.
   * This allows reusing the same iterator instance for multiple conversations.
   *
   * @param state The new conversation state to process
   */
  public void setState(ConversationState state) {
    this.currentState = state;
  }

  /**
   * Gets the current conversation state being processed.
   */
  public ConversationState getState() {
    return currentState;
  }

  /**
   * Creates a stream that generates tokens until a finish condition is met.
   */
  public Stream<T> stream() {
    return StreamSupport.stream(Spliterators.spliteratorUnknownSize(this, Spliterator.ORDERED), false);
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

    // Create batch for the prompt (using the sequence-aware constructor)
    LlamaBatch promptBatch = new LlamaBatch(arena, state.getTokenized(), state.getSequenceId());

    // Decode the batch
    if (promptBatch.decode(context) != 0) {
      promptBatch.free();
      throw new LlamaException("Failed to decode prompt for sequence " + state.getSequenceId());
    }

    // Update nPast after processing prompt
    state.setNPast(state.getTokenized().size());

    // Sample first token
    Integer newToken = sampler.sample(context);
    String tokenPiece = tokenizer.tokenToPiece(newToken);

    // Update state evaluation
    GenerationState newState = state
      .getStateEvaluation()
      .evaluate(new io.gravitee.llama.cpp.modules.StateEvaluation.Context(state.getGenerationState(), tokenPiece));
    state.setGenerationState(newState);

    if (newState == GenerationState.TOOLS) {
      state.setFinishReason(FinishReason.TOOL_CALL);
    }

    // Track tokens
    state.getTokenTracking().consume(new io.gravitee.llama.cpp.modules.TokenTracking.Context(state.getGenerationState(), 1));

    // Check if finished immediately
    if (!tokenizer.isEog(newToken)) {
      // Not finished - set up for next iteration
      state.setNewTokenId(newToken);
      state.setPiece(tokenPiece);
    } else {
      // Finished immediately
      state.setFinishReason(FinishReason.STOP);
    }

    promptBatch.free();
  }

  /**
   * Processes a sampled token for a given state.
   * Updates state evaluation, checks for tool calls, and tracks tokens.
   *
   * @param state The conversation state to update
   * @param tokenPiece The token piece that was sampled
   */
  protected void processSampledToken(ConversationState state, String tokenPiece) {
    // Update state evaluation
    GenerationState newState = state
      .getStateEvaluation()
      .evaluate(new io.gravitee.llama.cpp.modules.StateEvaluation.Context(state.getGenerationState(), tokenPiece));
    state.setGenerationState(newState);

    // Check for tool call
    if (newState == GenerationState.TOOLS) {
      state.setFinishReason(FinishReason.TOOL_CALL);
    }

    // Track tokens
    state.getTokenTracking().consume(new io.gravitee.llama.cpp.modules.TokenTracking.Context(state.getGenerationState(), 1));
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
      state.setFinishReason(FinishReason.STOP);
      return false;
    }

    // Check token limit
    int maxTokens = state.getMaxTokens();
    if (maxTokens != -1 && maxTokens <= state.getAnswerTokens()) {
      state.setFinishReason(FinishReason.LENGTH);
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
    boolean hasNotReachedQuota = maxTokens == -1 || maxTokens > currentState.getAnswerTokens();
    if (!hasNotReachedQuota) {
      setFinishReason(LENGTH);
    }
    return hasNotReachedQuota;
  }

  protected boolean endWithStopString() {
    if (!currentState.getPromptMemory().isInitialized()) {
      return false;
    }

    boolean endsWithStopString = currentState.getStopString().evaluate(currentState.getPromptMemory().getMemory());
    if (endsWithStopString) {
      setFinishReason(STOP);
    }
    return endsWithStopString;
  }

  protected void setFinishReason(FinishReason finishReason) {
    if (currentState.getFinishReason() != null) {
      if (!TOOL_CALL.equals(currentState.getFinishReason()) || LENGTH.equals(finishReason)) {
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

  protected void incrementTokenCount(int tokenCount) {
    currentState
      .getTokenTracking()
      .consume(new io.gravitee.llama.cpp.modules.TokenTracking.Context(currentState.getGenerationState(), tokenCount));
  }

  /**
   * Called when iteration completes.
   * Automatically cleans up the sequence from KV cache.
   */
  protected void onFinished() {
    if (currentState.getFinishReason() != null) {
      currentState.getContext().getMemory().seqRm(currentState.getSequenceId(), -1, -1);
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
