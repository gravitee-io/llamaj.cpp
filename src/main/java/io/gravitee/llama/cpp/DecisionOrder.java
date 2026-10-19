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
 * The role of a prompt token for a joint decision head ({@link DecisionType#CLEF}), mirroring
 * llama.cpp's staging {@code enum llama_decision_order}.
 * <p>
 * A run of tokens with the same value is one span; spans must be separated by {@link #NONE}
 * tokens. An option span belongs to the last question span before it.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public enum DecisionOrder {
  /** Not read by the head. */
  NONE(0),
  /** Text of a yes/no question. */
  QUESTION_NOUL(1),
  /** Text of a choice question. */
  QUESTION_CHOICE(2),
  /** Text of a score question. */
  QUESTION_SCORE(3),
  /** Text of an option. */
  OPTION(4);

  private final int nativeValue;

  DecisionOrder(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  /** The native {@code llama_decision_order} value. */
  public int nativeValue() {
    return nativeValue;
  }
}
