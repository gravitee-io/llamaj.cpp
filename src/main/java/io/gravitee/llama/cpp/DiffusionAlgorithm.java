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
 * Strategy used by a diffusion model to score how confident it is in each freshly
 * sampled token, which in turn decides the order in which masked positions are
 * unmasked ("transferred") across denoising steps.
 *
 * <p>Mirrors {@code enum diffusion_algorithm} in llama.cpp's diffusion example; the
 * ordinals match the native values.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public enum DiffusionAlgorithm {
  /** Confidence = probability of the sampled token (same scoring as CONFIDENCE_BASED). */
  ORIGIN,
  /** Confidence = negative entropy of the distribution (lower entropy = more confident). */
  ENTROPY_BASED,
  /** Confidence = margin between the top-1 and top-2 token probabilities. */
  MARGIN_BASED,
  /** Confidence drawn uniformly at random. */
  RANDOM,
  /** Confidence = probability of the sampled token. The default. */
  CONFIDENCE_BASED;

  public int nativeValue() {
    return ordinal();
  }

  public static DiffusionAlgorithm fromOrdinal(int ordinal) {
    return values()[ordinal];
  }
}
