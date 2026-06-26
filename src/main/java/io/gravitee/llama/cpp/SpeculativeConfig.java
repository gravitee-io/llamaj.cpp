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
 * Configuration for speculative decoding (see {@link ConversationState#setDraft}).
 *
 * <p>Only memoryless sampling is supported, because rejection sampling is exact only for it:
 * {@code temperature}, {@code topK}, {@code topP}. {@code temperature == 0} selects the
 * greedy (argmax) path, which is lossless w.r.t. greedy decoding. Stateful/reshaping
 * samplers (penalties, grammar, mirostat) are intentionally not supported here.
 *
 * <p><b>Adaptive draft length.</b> {@code nDraft} is the <i>maximum</i> tokens drafted per round.
 * When {@code pMin > 0}, the draft stops early as soon as its own top-token probability falls
 * below {@code pMin} (after at least {@code draftMin} tokens) — tokens drafted past the point
 * where the draft is unsure are usually rejected by the target and waste both draft and target
 * compute. Stopping early never changes <i>which</i> tokens are emitted (the target still verifies
 * every drafted token and always commits at least one), only <i>how many</i> are speculated, so it
 * preserves greedy losslessness and rejection-sampling exactness. {@code pMin <= 0} (the default)
 * disables early stop, always drafting exactly {@code nDraft} tokens.
 *
 * @param nDraft      Maximum tokens the draft proposes per round (K), must be >= 1
 * @param temperature Sampling temperature; {@code 0} = greedy
 * @param topK        Top-K cutoff ({@code <= 0} disables)
 * @param topP        Top-P / nucleus cutoff ({@code >= 1} disables)
 * @param seed        RNG seed for draft sampling, the accept coin, and residual draws
 * @param draftMin    Minimum tokens to draft before the {@code pMin} early-stop applies
 *                    (clamped to {@code [1, nDraft]})
 * @param pMin        Draft-confidence floor: stop drafting once the draft's top-token probability
 *                    drops below this ({@code <= 0} disables adaptive early stop)
 * @param ngram       N-gram (prompt-lookup) drafting: {@code 0} = use a draft model (see
 *                    {@link ConversationState#setDraft}); {@code >= 1} = draft tokens by matching the
 *                    last {@code ngram} committed tokens against the generation history (no draft
 *                    model, no draft forward pass — see {@link ConversationState#setNgram}). The
 *                    target still verifies every proposed token, so it stays lossless/exact.
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record SpeculativeConfig(
  int nDraft,
  float temperature,
  int topK,
  float topP,
  long seed,
  int draftMin,
  float pMin,
  int ngram
) {
  public SpeculativeConfig {
    if (nDraft < 1) {
      throw new LlamaException("nDraft must be >= 1");
    }
    if (ngram < 0) {
      throw new LlamaException("ngram must be >= 0");
    }
    // Keep draftMin in [1, nDraft] so a round always drafts at least one and at most nDraft tokens.
    draftMin = Math.max(1, Math.min(draftMin, nDraft));
  }

  /** Model-draft config (no n-gram) with explicit adaptive parameters. */
  public SpeculativeConfig(
    int nDraft,
    float temperature,
    int topK,
    float topP,
    long seed,
    int draftMin,
    float pMin
  ) {
    this(nDraft, temperature, topK, topP, seed, draftMin, pMin, 0);
  }

  /**
   * Backward-compatible fixed-length model-draft config: always drafts exactly {@code nDraft}
   * tokens (no adaptive early stop, no n-gram).
   */
  public SpeculativeConfig(
    int nDraft,
    float temperature,
    int topK,
    float topP,
    long seed
  ) {
    this(nDraft, temperature, topK, topP, seed, nDraft, 0.0f);
  }

  /** Greedy (argmax) model-draft speculative decoding with the given fixed draft window. */
  public static SpeculativeConfig greedy(int nDraft) {
    return new SpeculativeConfig(nDraft, 0.0f, 0, 1.0f, 0L);
  }

  /**
   * Greedy (argmax) model-draft speculative decoding with adaptive draft length: drafts between
   * {@code draftMin} and {@code draftMax} tokens per round, stopping early when the draft's
   * top-token probability drops below {@code pMin}. Lossless w.r.t. greedy decoding.
   */
  public static SpeculativeConfig greedyAdaptive(
    int draftMax,
    int draftMin,
    float pMin
  ) {
    return new SpeculativeConfig(draftMax, 0.0f, 0, 1.0f, 0L, draftMin, pMin);
  }

  /**
   * Greedy n-gram (prompt-lookup) drafting: propose up to {@code kMax} tokens by matching the last
   * {@code ngram} committed tokens against the generation history. No draft model. Lossless w.r.t.
   * greedy decoding (the target verifies every proposed token).
   */
  public static SpeculativeConfig ngramGreedy(int kMax, int ngram) {
    return new SpeculativeConfig(kMax, 0.0f, 0, 1.0f, 0L, kMax, 0.0f, ngram);
  }

  /**
   * Sampling n-gram (prompt-lookup) drafting: same lookup as {@link #ngramGreedy} but with
   * temperature/top-k/top-p. Stays an exact sampler of the target distribution (the proposed token
   * is treated as a point-mass draft for rejection sampling).
   */
  public static SpeculativeConfig ngram(
    int kMax,
    int ngram,
    float temperature,
    int topK,
    float topP,
    long seed
  ) {
    return new SpeculativeConfig(
      kMax,
      temperature,
      topK,
      topP,
      seed,
      kMax,
      0.0f,
      ngram
    );
  }

  public boolean isGreedy() {
    return temperature <= 0.0f;
  }

  /** Whether confidence-gated early stop is enabled ({@code pMin > 0}). Model-draft only. */
  public boolean isAdaptive() {
    return pMin > 0.0f && ngram == 0;
  }

  /** Whether this is n-gram (prompt-lookup) drafting rather than model drafting. */
  public boolean isNgram() {
    return ngram > 0;
  }

  /*
   * Fluent "wither" API: start from a factory preset and override individual settings, e.g.
   *
   *   SpeculativeConfig.greedy(8)
   *     .withTemperature(0.8f)   // turns the preset into a sampling config
   *     .withTopK(40)
   *     .withTopP(0.95f)
   *     .withSeed(42);
   *
   * Each call returns a new immutable config (the canonical constructor re-validates).
   */

  /** A copy with a different max draft window (K / kMax). */
  public SpeculativeConfig withNDraft(int nDraft) {
    return new SpeculativeConfig(
      nDraft,
      temperature,
      topK,
      topP,
      seed,
      draftMin,
      pMin,
      ngram
    );
  }

  /** A copy with a different sampling temperature ({@code 0} = greedy). */
  public SpeculativeConfig withTemperature(float temperature) {
    return new SpeculativeConfig(
      nDraft,
      temperature,
      topK,
      topP,
      seed,
      draftMin,
      pMin,
      ngram
    );
  }

  /** A copy with a different top-k cutoff ({@code <= 0} disables). */
  public SpeculativeConfig withTopK(int topK) {
    return new SpeculativeConfig(
      nDraft,
      temperature,
      topK,
      topP,
      seed,
      draftMin,
      pMin,
      ngram
    );
  }

  /** A copy with a different top-p / nucleus cutoff ({@code >= 1} disables). */
  public SpeculativeConfig withTopP(float topP) {
    return new SpeculativeConfig(
      nDraft,
      temperature,
      topK,
      topP,
      seed,
      draftMin,
      pMin,
      ngram
    );
  }

  /** A copy with a different RNG seed. */
  public SpeculativeConfig withSeed(long seed) {
    return new SpeculativeConfig(
      nDraft,
      temperature,
      topK,
      topP,
      seed,
      draftMin,
      pMin,
      ngram
    );
  }

  /** A copy with a different adaptive {@code draftMin} (clamped to {@code [1, nDraft]}). */
  public SpeculativeConfig withDraftMin(int draftMin) {
    return new SpeculativeConfig(
      nDraft,
      temperature,
      topK,
      topP,
      seed,
      draftMin,
      pMin,
      ngram
    );
  }

  /** A copy with a different adaptive confidence floor {@code pMin} ({@code <= 0} disables). */
  public SpeculativeConfig withPMin(float pMin) {
    return new SpeculativeConfig(
      nDraft,
      temperature,
      topK,
      topP,
      seed,
      draftMin,
      pMin,
      ngram
    );
  }

  /** A copy with a different n-gram lookup window ({@code 0} = model drafting, {@code >= 1} = n-gram). */
  public SpeculativeConfig withNgram(int ngram) {
    return new SpeculativeConfig(
      nDraft,
      temperature,
      topK,
      topP,
      seed,
      draftMin,
      pMin,
      ngram
    );
  }

  /** Default draft window used by {@link #builder()}. */
  public static final int DEFAULT_N_DRAFT = 4;

  /**
   * A fresh {@link Builder} with greedy model-draft defaults (fixed window of
   * {@value #DEFAULT_N_DRAFT}, no sampling, no n-gram).
   */
  public static Builder builder() {
    return new Builder();
  }

  /** A {@link Builder} seeded from this config's current settings. */
  public Builder toBuilder() {
    return new Builder()
      .nDraft(nDraft)
      .temperature(temperature)
      .topK(topK)
      .topP(topP)
      .seed(seed)
      .draftMin(draftMin)
      .pMin(pMin)
      .ngram(ngram);
  }

  /**
   * Fluent builder for {@link SpeculativeConfig}. Start from {@link #builder()} (greedy model-draft
   * defaults) or {@link #toBuilder()} (an existing config), set what you need, then {@link #build()}
   * — which validates through the record's canonical constructor.
   *
   * <pre>{@code
   * var config = SpeculativeConfig.builder()
   *     .nDraft(8)
   *     .temperature(0.8f).topK(40).topP(0.95f).seed(42)
   *     .build();
   *
   * // n-gram (prompt-lookup) drafting:
   * var ng = SpeculativeConfig.builder().nDraft(4).ngram(2).build();
   * }</pre>
   */
  public static final class Builder {

    private int nDraft = DEFAULT_N_DRAFT;
    private float temperature = 0.0f;
    private int topK = 0;
    private float topP = 1.0f;
    private long seed = 0L;
    private int draftMin = 1;
    private float pMin = 0.0f;
    private int ngram = 0;

    private Builder() {}

    /** Maximum tokens drafted/proposed per round (K / kMax). */
    public Builder nDraft(int nDraft) {
      this.nDraft = nDraft;
      return this;
    }

    /** Sampling temperature ({@code 0} = greedy). */
    public Builder temperature(float temperature) {
      this.temperature = temperature;
      return this;
    }

    /** Top-k cutoff ({@code <= 0} disables). */
    public Builder topK(int topK) {
      this.topK = topK;
      return this;
    }

    /** Top-p / nucleus cutoff ({@code >= 1} disables). */
    public Builder topP(float topP) {
      this.topP = topP;
      return this;
    }

    /** RNG seed for draft sampling, the accept coin, and residual draws. */
    public Builder seed(long seed) {
      this.seed = seed;
      return this;
    }

    /** Minimum tokens to draft before the {@code pMin} early-stop applies (model-draft only). */
    public Builder draftMin(int draftMin) {
      this.draftMin = draftMin;
      return this;
    }

    /** Adaptive confidence floor; stop drafting below it ({@code <= 0} disables, model-draft only). */
    public Builder pMin(float pMin) {
      this.pMin = pMin;
      return this;
    }

    /** N-gram lookup window ({@code 0} = model drafting, {@code >= 1} = n-gram prompt-lookup). */
    public Builder ngram(int ngram) {
      this.ngram = ngram;
      return this;
    }

    /** Builds the immutable config (validated by the record's canonical constructor). */
    public SpeculativeConfig build() {
      return new SpeculativeConfig(
        nDraft,
        temperature,
        topK,
        topP,
        seed,
        draftMin,
        pMin,
        ngram
      );
    }
  }
}
