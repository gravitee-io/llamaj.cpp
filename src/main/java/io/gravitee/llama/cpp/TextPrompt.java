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
 * A prompt for {@link LabelDecider} or {@link KevDecider}: text (tokenized by the decider with
 * special-token parsing and without an added BOS) or tokens.
 * <ul>
 *   <li>{@link LabelDecider}: scores the first {@code nOptions} labels</li>
 *   <li>{@link KevDecider}: the prompt must hold {@code nOptions} option markers</li>
 * </ul>
 *
 * @param text     The prompt text, or {@code null} if {@code tokens} is given
 * @param tokens   The prompt tokens, or {@code null} if {@code text} is given
 * @param nOptions The number of scores to return
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record TextPrompt(String text, int[] tokens, int nOptions) implements
  DecisionPrompt {
  public TextPrompt {
    if ((text == null) == (tokens == null)) {
      throw new IllegalArgumentException("give either text or tokens");
    }
    if (nOptions < 1) {
      throw new IllegalArgumentException("nOptions must be at least 1");
    }
  }

  /** A text prompt. */
  public static TextPrompt of(String text, int nOptions) {
    return new TextPrompt(text, null, nOptions);
  }

  /** A tokenized prompt. */
  public static TextPrompt of(int[] tokens, int nOptions) {
    return new TextPrompt(null, tokens, nOptions);
  }
}
