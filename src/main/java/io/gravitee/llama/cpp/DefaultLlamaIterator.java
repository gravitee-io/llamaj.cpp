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

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class DefaultLlamaIterator extends LlamaIterator {

  private LlamaBatch batch;
  private Integer newTokenId;

  public DefaultLlamaIterator(Arena arena, LlamaContext context, LlamaTokenizer tokenizer, LlamaSampler sampler) {
    super(arena, context, tokenizer, sampler);
  }

  @Override
  public boolean hasNext() {
    return hasNext;
  }

  public boolean batch() {
    batch = newTokenId == null ? new LlamaBatch(arena, tokenized) : new LlamaBatch(arena, newTokenId);

    if (checkContextSize() && batch.decode(context) != 0) {
      return false;
    }

    newTokenId = sampler.sample(context);

    batch.free();
    batch = null;

    incrementTokenCount(1);
    return hasNotReachedQuota() && !isEog(newTokenId);
  }

  private boolean checkContextSize() {
    return context.nCtxUsedCells() + batch.nTokens() <= context.nCtx();
  }

  @Override
  public LlamaOutput next() {
    var piece = tokenizer.tokenToPiece(newTokenId);

    feedPromptMemory(piece);

    hasNext = !endWithStopString() && batch();
    return new LlamaOutput(piece, 1);
  }
}
