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
 * Schedule controlling how many masked positions are unmasked at each denoising step.
 *
 * <p>Mirrors {@code enum diffusion_transfer_schedule} in llama.cpp's diffusion example;
 * the ordinals match the native values.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public enum DiffusionTransferSchedule {
  /** Dream-style: {@code (1.0 - s/t) * remaining} over the whole sequence. */
  TIMESTEP_BASED,
  /** LLaDA-style: process the sequence in fixed-size blocks. Requires a block length. */
  BLOCK_BASED;

  public int nativeValue() {
    return ordinal();
  }

  public static DiffusionTransferSchedule fromOrdinal(int ordinal) {
    return values()[ordinal];
  }
}
