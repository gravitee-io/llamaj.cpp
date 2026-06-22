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
 * Configuration for {@link DiffusionGenerator}. Mirrors the {@code diffusion_params}
 * struct from llama.cpp's diffusion example, but is a plain Java holder since that
 * struct lives in the example layer and is not part of the public C ABI.
 *
 * <p>Defaults match a sensible Dream-style (timestep-based, confidence) configuration.
 * {@code maskTokenId} and {@code maxLength} have no meaningful default and must be set.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class DiffusionParams {

  /** Mask/null token sentinel used by llama.cpp ({@code LLAMA_TOKEN_NULL}). */
  public static final int LLAMA_TOKEN_NULL = -1;

  private int steps = 64;
  private float temperature = 0.0f;
  private int maskTokenId = LLAMA_TOKEN_NULL;
  private int seed = 0;
  private Boolean shiftLogits = null;

  private float topP = 1.0f;
  private int topK = 0;

  private DiffusionAlgorithm algorithm = DiffusionAlgorithm.CONFIDENCE_BASED;
  private DiffusionTransferSchedule schedule =
    DiffusionTransferSchedule.TIMESTEP_BASED;

  private float eps = 1e-3f;
  private int blockLength = 0;
  private float algTemp = 0.0f;
  private boolean addGumbelNoise = false;

  private int maxLength = 0;

  public int steps() {
    return steps;
  }

  public DiffusionParams steps(int steps) {
    this.steps = steps;
    return this;
  }

  public float temperature() {
    return temperature;
  }

  public DiffusionParams temperature(float temperature) {
    this.temperature = temperature;
    return this;
  }

  public int maskTokenId() {
    return maskTokenId;
  }

  public DiffusionParams maskTokenId(int maskTokenId) {
    this.maskTokenId = maskTokenId;
    return this;
  }

  public int seed() {
    return seed;
  }

  public DiffusionParams seed(int seed) {
    this.seed = seed;
    return this;
  }

  /**
   * Explicit {@code shift_logits} override, or {@code null} to auto-resolve from the model's
   * GGUF metadata / architecture (see {@code DiffusionCanvasState.resolveShiftLogits}).
   */
  public Boolean shiftLogits() {
    return shiftLogits;
  }

  /** Forces {@code shift_logits}; pass {@code null} to fall back to model auto-detection. */
  public DiffusionParams shiftLogits(Boolean shiftLogits) {
    this.shiftLogits = shiftLogits;
    return this;
  }

  public float topP() {
    return topP;
  }

  public DiffusionParams topP(float topP) {
    this.topP = topP;
    return this;
  }

  public int topK() {
    return topK;
  }

  public DiffusionParams topK(int topK) {
    this.topK = topK;
    return this;
  }

  public DiffusionAlgorithm algorithm() {
    return algorithm;
  }

  public DiffusionParams algorithm(DiffusionAlgorithm algorithm) {
    this.algorithm = algorithm;
    return this;
  }

  public DiffusionTransferSchedule schedule() {
    return schedule;
  }

  public DiffusionParams schedule(DiffusionTransferSchedule schedule) {
    this.schedule = schedule;
    return this;
  }

  public float eps() {
    return eps;
  }

  public DiffusionParams eps(float eps) {
    this.eps = eps;
    return this;
  }

  public int blockLength() {
    return blockLength;
  }

  public DiffusionParams blockLength(int blockLength) {
    this.blockLength = blockLength;
    return this;
  }

  public float algTemp() {
    return algTemp;
  }

  public DiffusionParams algTemp(float algTemp) {
    this.algTemp = algTemp;
    return this;
  }

  public boolean addGumbelNoise() {
    return addGumbelNoise;
  }

  public DiffusionParams addGumbelNoise(boolean addGumbelNoise) {
    this.addGumbelNoise = addGumbelNoise;
    return this;
  }

  public int maxLength() {
    return maxLength;
  }

  public DiffusionParams maxLength(int maxLength) {
    this.maxLength = maxLength;
    return this;
  }
}
