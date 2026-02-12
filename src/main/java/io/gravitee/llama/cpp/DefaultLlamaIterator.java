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

/**
 * Default implementation of LlamaIterator that processes conversations token by token.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class DefaultLlamaIterator extends LlamaIterator<LlamaOutput> {

  public DefaultLlamaIterator(
    ConversationState initialState,
    MtmdContext mtmdContext
  ) {
    super(initialState, mtmdContext);
  }

  public DefaultLlamaIterator(ConversationState initialState) {
    this(initialState, null);
  }

  @Override
  protected boolean batch() {
    var arena = currentState.getArena();
    var context = currentState.getContext();
    var sampler = currentState.getSampler();

    LlamaBatch batch;
    if (currentState.getNewTokenId() == null) {
      // Initial prompt processing - use shared method
      processPrompt(currentState);
      feedPromptMemory(currentState.getPiece());
      return (
        currentState.getFinishReason() == null &&
        !endWithStopString() &&
        hasNotReachedQuota()
      );
    } else {
      // Single token generation - need to specify position and sequence ID
      batch = new LlamaBatch(arena, 1, 0, 1);
      batch.add(
        currentState.getNewTokenId(),
        currentState.getNPast(),
        java.util.List.of(currentState.getSequenceId()),
        true
      );
    }

    if (checkContextSize(batch) && batch.decode(context) != 0) {
      setFinishReason(FinishReason.STOP);
      batch.free();
      return false;
    }

    // After single token: increment nPast
    currentState.incrementNPast();

    int newToken = sampler.sample(context);
    String tokenPiece = decodeTokenPiece(currentState, newToken);

    // Process the sampled token using shared helper method
    processSampledToken(currentState, tokenPiece);

    if (isEog(newToken)) {
      incrementTokenCount(-1);
      batch.free();
      return false;
    }

    batch.free();

    currentState.setNewTokenId(newToken);
    currentState.setPiece(tokenPiece);

    feedPromptMemory(tokenPiece);
    return !endWithStopString() && hasNotReachedQuota();
  }

  private boolean checkContextSize(LlamaBatch batch) {
    return (
      currentState.getContext().nCtxUsedCells() + batch.nTokens() <=
      currentState.getContext().nCtx()
    );
  }

  @Override
  public LlamaOutput next() {
    return new LlamaOutput(currentState.getPiece(), 1);
  }
}
