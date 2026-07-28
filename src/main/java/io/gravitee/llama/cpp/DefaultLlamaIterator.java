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

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Default implementation of LlamaIterator that processes conversations token by token.
 *
 * <p>When the {@link ConversationState} has a draft context configured
 * ({@link ConversationState#setDraft}), it runs greedy speculative decoding instead:
 * a draft → verify → accept cycle producing a burst of tokens per step, drained one per
 * {@link #next()} via {@link #pending}.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class DefaultLlamaIterator
  extends LlamaIterator<LlamaOutput>
  implements AutoCloseable {

  // Accepted speculative tokens awaiting emission (speculative mode only).
  private final Deque<LlamaOutput> pending = new ArrayDeque<>();

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
    if (currentState.isSpeculative()) {
      return speculativeBatch();
    }
    if (currentState.isFinished()) {
      // A previous step finished generation but emitted a final flushed marker prefix.
      return false;
    }
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

    if (!checkContextSize(batch)) {
      // Context is full: decoding another token is impossible. Stop here,
      // otherwise we would skip the decode and keep re-sampling stale logits
      // in an infinite loop.
      setFinishReason(FinishReason.LENGTH);
      batch.free();
      return false;
    }

    if (batch.decode(context) != 0) {
      setFinishReason(FinishReason.STOP);
      batch.free();
      return false;
    }

    // After single token: increment nPast — the decoded token's KV row is now resident.
    currentState.incrementNPast();
    currentState.getTokenHistory().append(currentState.getNewTokenId());

    // Budget-aware EOG boost, written into the logits this sample() is about to read.
    // -1 is the last output row, which is what sample(context) reads.
    applyEogRamp(currentState, -1);
    int newToken = sampler.sample(context);
    String tokenPiece = decodeTokenPiece(currentState, newToken);

    // Collect logprobs before processing (logits are invalidated after next decode).
    Logprobs logprobs = collectLogprobs(currentState, newToken, -1);

    if (isEog(newToken)) {
      batch.free();
      currentState.setFinished(true);
      // Generation ended while a multi-token marker prefix may still be buffered — flush it
      // to the current channel as one final emission instead of dropping it.
      var flush = currentState
        .getStateEvaluation()
        .flushPending(currentState.getGenerationState());
      if (flush.emitTokens() > 0) {
        incrementTokenCount(flush.emitTokens());
        currentState.setPiece(flush.emit());
        currentState.setPieceTokens(flush.emitTokens());
        currentState.setLogprobs(null);
        return true;
      }
      return false;
    }

    // Process the sampled token using shared helper method (token-sequence aware)
    var emission = processSampledToken(currentState, newToken, tokenPiece);

    batch.free();

    currentState.setNewTokenId(newToken);
    currentState.setPiece(emission.emit());
    currentState.setPieceTokens(emission.emitTokens());
    currentState.setLogprobs(logprobs);

    feedPromptMemory(tokenPiece);
    return !endWithStopString() && hasNotReachedQuota();
  }

  private boolean checkContextSize(LlamaBatch batch) {
    return (
      currentState.getContext().nCtxUsedCells() + batch.nTokens() <=
      currentState.getContext().nCtx()
    );
  }

  /** Speculative-mode step: prompt prefill (target + draft) + first token, then accept bursts. */
  private boolean speculativeBatch() {
    if (!pending.isEmpty()) {
      return true;
    }
    if (currentState.isFinished()) {
      return false;
    }
    if (currentState.getNewTokenId() == null) {
      processPrompt(currentState);
      prefillDraft(currentState); // no-op for n-gram (no draft context)
      feedPromptMemory(currentState.getPiece());
      if (currentState.getFinishReason() == null) {
        if (currentState.isNgram()) {
          currentState.seedNgramHistory();
        }
        if (
          currentState.getPiece() != null && !currentState.getPiece().isEmpty()
        ) {
          pending.add(
            new LlamaOutput(
              currentState.getPiece(),
              currentState.getPieceTokens(),
              currentState.getLogprobs()
            )
          );
        }
      } else {
        currentState.setFinished(true);
      }
      if (!pending.isEmpty()) {
        return true;
      }
      // fall through: the first token was buffered as a marker prefix — run rounds until
      // something is emitted or generation finishes.
    }
    // A round may emit nothing while a multi-token marker prefix is being buffered; keep
    // going until something is emitted or the state finishes (each round commits >= 1 token,
    // so this terminates).
    while (pending.isEmpty() && !currentState.isFinished()) {
      pending.addAll(speculativeRound(currentState));
    }
    return !pending.isEmpty();
  }

  @Override
  public LlamaOutput next() {
    if (currentState.isSpeculative()) {
      return pending.poll();
    }
    return new LlamaOutput(
      currentState.getPiece(),
      currentState.getPieceTokens(),
      currentState.getLogprobs()
    );
  }

  /**
   * Releases this conversation's resources: frees the speculative state's persistent native scratch
   * (if any) and removes its sequence from the context KV cache. Idempotent (null-on-free + a no-op
   * seqRm), so it is safe whether the stream ran to completion or was abandoned early — call it via
   * try-with-resources when not consuming the whole stream.
   */
  @Override
  public void close() {
    if (currentState.isSpeculative()) {
      currentState.freeSpeculativeScratch();
    }
    if (!currentState.isRetainKv()) {
      currentState
        .getContext()
        .getMemory()
        .seqRm(currentState.getSequenceId(), -1, -1);
    }
  }
}
