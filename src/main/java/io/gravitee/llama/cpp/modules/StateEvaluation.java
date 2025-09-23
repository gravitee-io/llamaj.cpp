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

import static io.gravitee.llama.cpp.GenerationState.TOOLS;
import static io.gravitee.llama.cpp.modules.StateEvaluation.*;
import static java.util.function.Function.identity;
import static java.util.function.Predicate.not;
import static java.util.stream.Collectors.toMap;

import io.gravitee.llama.cpp.GenerationState;
import io.gravitee.llama.cpp.StateBounds;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class StateEvaluation implements Initializable<Config>, Evaluable<Context, GenerationState> {

  private Map<GenerationState, StateBounds> states;
  private Map<GenerationState, Boolean> occurredState;

  @Override
  public boolean isInitialized() {
    return states != null && !states.isEmpty();
  }

  @Override
  public GenerationState evaluate(Context context) {
    return isInitialized()
      ? switch (context.currentState) {
        case ANSWER -> detectNewState(context.piece);
        case REASONING, TOOLS -> detectEndState(context.currentState, context.piece);
        case null -> GenerationState.ANSWER;
      }
      : GenerationState.ANSWER;
  }

  private GenerationState detectEndState(GenerationState currentState, String piece) {
    var state = states.get(currentState);

    if (stateAlreadyOccurred(state)) {
      return GenerationState.ANSWER;
    }

    if (state.end().equals(piece)) {
      setAlreadyOccurredIfNecessary(currentState);
      return GenerationState.ANSWER;
    }

    return currentState;
  }

  private void setAlreadyOccurredIfNecessary(GenerationState currentState) {
    boolean isTools = !TOOLS.equals(currentState);
    this.occurredState.put(currentState, isTools);
  }

  private GenerationState detectNewState(String piece) {
    return states
      .entrySet()
      .stream()
      .filter(not(e -> this.stateAlreadyOccurred(e.getValue())))
      .map(Entry::getValue)
      .filter(stateBounds -> stateBounds.start().equals(piece))
      .map(StateBounds::state)
      .findFirst()
      .orElse(GenerationState.ANSWER);
  }

  private Boolean stateAlreadyOccurred(StateBounds stateBounds) {
    if (stateBounds == null) {
      return true;
    }

    if (TOOLS.equals(stateBounds.state())) {
      return false;
    }

    return occurredState.get(stateBounds.state());
  }

  @Override
  public void initialize(Config config) {
    this.states = config.states.stream().collect(toMap(StateBounds::state, identity()));
    this.occurredState = this.states.keySet().stream().collect(toMap(identity(), __ -> false));
  }

  public record Config(List<StateBounds> states) {}

  public record Context(GenerationState currentState, String piece) {}
}
