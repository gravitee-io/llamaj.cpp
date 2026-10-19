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
 * The kind of question a {@link DecisionPrompt} asks. Some decision families read it: laya picks
 * the output column of its head from it ({@link #column()}), clef tags question spans with it
 * ({@link DecisionOrder}).
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public enum QuestionType {
  /** Pick one option out of named options. */
  CHOICE(0),
  /** Rate on an ordered scale. */
  SCORE(1),
  /** Yes or no. */
  NOUL(2);

  private final int column;

  QuestionType(int column) {
    this.column = column;
  }

  /** The output column of a laya head for this question type. */
  public int column() {
    return column;
  }
}
