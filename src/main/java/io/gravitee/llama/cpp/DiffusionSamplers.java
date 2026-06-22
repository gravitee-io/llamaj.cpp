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

import java.lang.foreign.Arena;

/**
 * Builds the sampler chain used by the diffusion generators from a {@link DiffusionParams}.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
final class DiffusionSamplers {

  private DiffusionSamplers() {}

  static LlamaSampler build(Arena arena, DiffusionParams params) {
    LlamaSampler sampler = new LlamaSampler(arena);
    if (params.topK() > 0) {
      sampler.topK(params.topK());
    }
    if (params.topP() < 1.0f) {
      sampler.topP(params.topP(), 1);
    }
    if (params.temperature() > 0.0f) {
      sampler.temperature(params.temperature());
    }
    sampler.seed(params.seed());
    return sampler;
  }
}
