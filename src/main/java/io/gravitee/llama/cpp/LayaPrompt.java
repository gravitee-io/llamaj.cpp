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

import java.util.Objects;

/**
 * A prompt for {@link LayaDecider}: tokens laid out the way the model was trained, holding
 * {@code nOptions} option markers, and the question type whose head column is read.
 *
 * @param tokens   The prompt tokens
 * @param type     The question type
 * @param nOptions The number of option markers, and of scores to return
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record LayaPrompt(
  int[] tokens,
  QuestionType type,
  int nOptions
) implements DecisionPrompt {
  public LayaPrompt {
    Objects.requireNonNull(tokens, "tokens");
    Objects.requireNonNull(type, "type");
    if (nOptions < 1) {
      throw new IllegalArgumentException("nOptions must be at least 1");
    }
  }
}
