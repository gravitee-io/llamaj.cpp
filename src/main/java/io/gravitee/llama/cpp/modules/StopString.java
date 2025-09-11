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

import static java.util.function.Predicate.not;

import java.util.List;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class StopString implements Initializable<List<String>>, Evaluable<String, Boolean> {

  private List<String> stopStrings;

  @Override
  public boolean isInitialized() {
    return stopStrings != null && !stopStrings.isEmpty();
  }

  @Override
  public void initialize(List<String> stopStrings) {
    if (stopStrings != null) {
      this.stopStrings = stopStrings.stream().filter(not(String::isBlank)).toList();
    }
  }

  @Override
  public Boolean evaluate(String input) {
    return isInitialized() && stopStrings.stream().anyMatch(input::endsWith);
  }
}
