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
 * On-demand tensor reading for model loading ({@code llama_lazy_mode}, llama.cpp v0.4.0+).
 * Lazy reading requires mmap; {@link #AUTO} is the native default.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public enum LlamaLazyMode {
  OFF(0), // always read the whole tensor up front
  AUTO(1), // lazy only for arch-marked tensors larger than 4 GiB
  ON(2); // read rows of arch-marked tensors on demand

  private final int value;

  LlamaLazyMode(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }

  public static LlamaLazyMode fromValue(int value) {
    for (LlamaLazyMode mode : values()) {
      if (mode.value == value) return mode;
    }
    throw new IllegalArgumentException("Unknown llama_lazy_mode: " + value);
  }
}
