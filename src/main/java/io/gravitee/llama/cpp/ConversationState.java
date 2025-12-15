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

import static io.gravitee.llama.cpp.GenerationState.ANSWER;

import io.gravitee.llama.cpp.LlamaTokenizer.TokenizerResponse;
import io.gravitee.llama.cpp.modules.PromptMemory;
import io.gravitee.llama.cpp.modules.StateEvaluation;
import io.gravitee.llama.cpp.modules.StopString;
import io.gravitee.llama.cpp.modules.TokenTracking;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;

/**
 * Encapsulates all state and resources for a single conversation/generation session.
 * The state owns the context, tokenizer, and sampler resources.
 *
 * <p>Example usage:
 * <pre>{@code
 * // Create conversation state with resources
 * var state = ConversationState.create(arena, context, tokenizer, sampler, sequenceId)
 *     .setMaxTokens(100)
 *     .initialize("Hello, how are you?");
 *
 * // Create iterator with this state
 * var iterator = new DefaultLlamaIterator(state);
 * iterator.stream().forEach(output -> System.out.print(output.text()));
 *
 * // Switch iterator to different state
 * var otherState = ConversationState.create(arena, context, tokenizer, sampler, 2)
 *     .initialize("Another conversation");
 * iterator.setState(otherState);
 * }</pre>
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class ConversationState {

  // Resources (owned by this state)
  private final Arena arena;
  private final LlamaContext context;
  private final LlamaTokenizer tokenizer;
  private final LlamaSampler sampler;

  // Identity & position
  private final int sequenceId;
  private int nPast = 0;

  // Tokenization
  private TokenizerResponse tokenized;

  // Tracking & state
  private final TokenTracking tokenTracking = new TokenTracking();
  private final PromptMemory promptMemory = new PromptMemory();
  private final StopString stopString = new StopString();
  private final StateEvaluation stateEvaluation = new StateEvaluation();

  // Generation state
  private GenerationState generationState = ANSWER;
  private FinishReason finishReason;

  // Configuration
  private int maxTokens = -1;
  private final List<StateBounds> stateBounds = new ArrayList<>();

  // Iteration state (used by iterator)
  Integer newTokenId;
  String piece;

  private ConversationState(
    Arena arena,
    LlamaContext context,
    LlamaTokenizer tokenizer,
    LlamaSampler sampler,
    int sequenceId
  ) {
    this.arena = arena;
    this.context = context;
    this.tokenizer = tokenizer;
    this.sampler = sampler;
    this.sequenceId = sequenceId;
  }

  /**
   * Creates a new conversation state with resources and sequence ID.
   *
   * @param arena The memory arena
   * @param context The LlamaContext to use
   * @param tokenizer The tokenizer to use
   * @param sampler The sampler to use
   * @param sequenceId The sequence ID for this conversation
   * @return A new conversation state
   */
  public static ConversationState create(
    Arena arena,
    LlamaContext context,
    LlamaTokenizer tokenizer,
    LlamaSampler sampler,
    int sequenceId
  ) {
    return new ConversationState(arena, context, tokenizer, sampler, sequenceId);
  }

  /**
   * Creates a new conversation state with default sequence ID (0).
   */
  public static ConversationState create(Arena arena, LlamaContext context, LlamaTokenizer tokenizer, LlamaSampler sampler) {
    return new ConversationState(arena, context, tokenizer, sampler, 0);
  }

  /**
   * Gets the arena used by this conversation.
   */
  public Arena getArena() {
    return arena;
  }

  /**
   * Initializes this conversation with a prompt.
   *
   * @param prompt The prompt text
   * @return This state for chaining
   */
  public ConversationState initialize(String prompt) {
    this.tokenized = tokenizer.tokenize(arena, prompt);
    this.tokenTracking.initialize(tokenized.size());
    this.stateEvaluation.initialize(new StateEvaluation.Config(stateBounds));
    this.generationState = ANSWER;
    this.finishReason = null;
    this.newTokenId = null;
    this.piece = null;
    this.nPast = 0;
    return this;
  }

  /**
   * Sets the maximum number of tokens to generate.
   */
  public ConversationState setMaxTokens(int maxTokens) {
    this.maxTokens = maxTokens;
    return this;
  }

  /**
   * Sets stop strings for this conversation.
   */
  public ConversationState setStopStrings(List<String> stopStrings) {
    this.stopString.initialize(stopStrings);
    int maxStringSize = stopStrings.stream().mapToInt(String::length).max().orElse(0);
    this.promptMemory.initialize(maxStringSize);
    return this;
  }

  /**
   * Configures reasoning token detection.
   */
  public ConversationState setReasoning(String tokenStart, String tokenEnd) {
    this.stateBounds.add(new StateBounds(GenerationState.REASONING, tokenStart, tokenEnd));
    return this;
  }

  /**
   * Configures tool call detection.
   */
  public ConversationState setToolCall(String tokenStart, String tokenEnd) {
    this.stateBounds.add(new StateBounds(GenerationState.TOOLS, tokenStart, tokenEnd));
    return this;
  }

  // Resource getters
  public LlamaContext getContext() {
    return context;
  }

  public LlamaTokenizer getTokenizer() {
    return tokenizer;
  }

  public LlamaSampler getSampler() {
    return sampler;
  }

  // State getters
  public int getSequenceId() {
    return sequenceId;
  }

  public int getNPast() {
    return nPast;
  }

  public void setNPast(int nPast) {
    this.nPast = nPast;
  }

  public void incrementNPast() {
    this.nPast++;
  }

  public TokenizerResponse getTokenized() {
    return tokenized;
  }

  public TokenTracking getTokenTracking() {
    return tokenTracking;
  }

  public PromptMemory getPromptMemory() {
    return promptMemory;
  }

  public StopString getStopString() {
    return stopString;
  }

  public StateEvaluation getStateEvaluation() {
    return stateEvaluation;
  }

  public GenerationState getGenerationState() {
    return generationState;
  }

  public void setGenerationState(GenerationState generationState) {
    this.generationState = generationState;
  }

  public FinishReason getFinishReason() {
    return finishReason;
  }

  public void setFinishReason(FinishReason finishReason) {
    this.finishReason = finishReason;
  }

  public int getMaxTokens() {
    return maxTokens;
  }

  public Integer getNewTokenId() {
    return newTokenId;
  }

  public void setNewTokenId(Integer newTokenId) {
    this.newTokenId = newTokenId;
  }

  public String getPiece() {
    return piece;
  }

  public void setPiece(String piece) {
    this.piece = piece;
  }

  // Token count accessors
  public int getInputTokens() {
    return tokenTracking.getInputTokenCount();
  }

  public int getAnswerTokens() {
    return tokenTracking.getOutputTokenCount(ANSWER);
  }

  public int getReasoningTokens() {
    return tokenTracking.getOutputTokenCount(GenerationState.REASONING);
  }

  public int getToolsTokens() {
    return tokenTracking.getOutputTokenCount(GenerationState.TOOLS);
  }

  public int getTotalTokenCount() {
    return tokenTracking.getTotalTokenCount();
  }
}
