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
package io.gravitee.llama.cpp.modules;

import io.gravitee.llama.cpp.GenerationState;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class TokenTracking implements Consumer<Integer, TokenTracking.Context> {

  private AtomicInteger input;
  private AtomicInteger reasoning;
  private AtomicInteger answer;
  private AtomicInteger toolCall;

  public void initialize(Integer initialTokenCount) {
    input = new AtomicInteger(initialTokenCount);
    answer = new AtomicInteger(0);
    reasoning = new AtomicInteger(0);
    toolCall = new AtomicInteger(0);
  }

  @Override
  public void consume(Context context) {
    switch (context.state) {
      case ANSWER -> answer.addAndGet(context.count);
      case REASONING -> reasoning.addAndGet(context.count);
      case TOOL_CALL -> toolCall.addAndGet(context.count);
    }
  }

  public record Context(GenerationState state, int count) {
    public Context(int count) {
      this(null, count);
    }
  }

  public int getInputTokenCount() {
    return input.get();
  }

  public int getOutputTokenCount(GenerationState state) {
    return switch (state) {
      case ANSWER -> answer.get();
      case REASONING -> reasoning.get();
      case TOOL_CALL -> toolCall.get();
      case null -> answer.get() + reasoning.get() + toolCall.get();
    };
  }

  public int getTotalTokenCount() {
    return getInputTokenCount() + getOutputTokenCount(null);
  }
}
