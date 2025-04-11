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

import java.util.Arrays;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public enum Role {
  ASSISTANT("assistant"),
  SYSTEM("system"),
  USER("user");

  private final String label;

  Role(String label) {
    this.label = label;
  }

  public String getLabel() {
    return label;
  }

  public static Role fromLabel(String label) {
    return Arrays
      .stream(Role.values())
      .filter(role -> role.label.equals(label))
      .findFirst()
      .orElseThrow(() -> new IllegalArgumentException("Invalid role: " + label));
  }
}
