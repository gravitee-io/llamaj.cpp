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

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

/**
 * Speculative-decoding sampling support shared by {@link LlamaIterator}'s single-sequence
 * round and {@link BatchIterator}'s fused step.
 *
 * <p>The per-position distribution (temperature → top-k → top-p → softmax) is computed
 * <b>natively</b> by the sampler chain applied to a {@link LlamaTokenDataArray}
 * ({@code llama_sampler_apply}); this class only adds the rejection-sampling decision and
 * residual draw, which have no native primitive. Reusable candidate buffers (one per role)
 * are allocated once and reused across rounds.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
final class Speculation {

  /** Post-sampler candidate distribution snapshot ({@code id -> prob} over the kept support). */
  record Snapshot(
    int[] ids,
    float[] probs,
    int selectedId,
    float selectedProbability
  ) {
    /** Highest probability over the kept support — the draft's confidence (for adaptive stop). */
    float maxProb() {
      float max = 0.0f;
      for (float p : probs) {
        if (p > max) {
          max = p;
        }
      }
      return max;
    }
  }

  private final SpeculativeConfig config;
  private final Arena arena;
  private final LlamaTokenDataArray targetBuf;
  private final LlamaTokenDataArray draftBuf;
  private final Random rng;
  // Reusable dense q-by-token-id buffer for the residual draw, kept all-zero between calls so a
  // residual is O(support) instead of O(support²) (the old per-id linear probOf scan).
  private final float[] qScatter;

  // Persistent native scratch, built once and reused every round (cleared in place), then freed by
  // free() exactly once on teardown. Reusing them avoids per-round native init/free and the
  // confined-arena growth from re-allocating these structs every round. Nullable / lazily built.
  private LlamaSampler chain;
  private LlamaBatch draftBatch;
  private LlamaBatch verifyBatch;

  Speculation(Arena arena, int nVocab, SpeculativeConfig config) {
    this.config = config;
    this.arena = arena;
    this.targetBuf = new LlamaTokenDataArray(arena, nVocab);
    this.draftBuf = new LlamaTokenDataArray(arena, nVocab);
    this.rng = new Random(config.seed());
    this.qScatter = new float[nVocab];
  }

  boolean isGreedy() {
    return config.isGreedy();
  }

  boolean isAdaptive() {
    return config.isAdaptive();
  }

  int draftMin() {
    return config.draftMin();
  }

  float pMin() {
    return config.pMin();
  }

  /**
   * Lazily-built persistent sampler chain for this config (greedy, or temp/top-k/top-p/dist),
   * reused across rounds. Reuse gives the stochastic chain a single continuous RNG stream (more
   * correct than re-seeding every round); the greedy chain is stateless. Freed by {@link #free()}.
   */
  LlamaSampler chain() {
    if (chain == null) {
      chain = buildChain();
    }
    return chain;
  }

  /** Builds the (memoryless) sampler chain for this config on the persistent arena. */
  private LlamaSampler buildChain() {
    LlamaSampler s = new LlamaSampler(arena);
    if (config.isGreedy()) {
      return s.greedy();
    }
    if (config.topK() > 0) {
      s.topK(config.topK());
    }
    if (config.topP() < 1.0f) {
      s.topP(config.topP(), 1);
    }
    s.temperature(config.temperature());
    s.seed((int) config.seed());
    return s;
  }

  /** Persistent single-token draft batch (reused via {@code clear()}). */
  LlamaBatch draftBatch() {
    if (draftBatch == null) {
      draftBatch = new LlamaBatch(arena, 1, 0, 1);
    }
    return draftBatch;
  }

  /** Persistent verify batch sized for idLast + up to {@code nDraft} drafted tokens. */
  LlamaBatch verifyBatch() {
    if (verifyBatch == null) {
      verifyBatch = new LlamaBatch(arena, config.nDraft() + 1, 0, 1);
    }
    return verifyBatch;
  }

  /**
   * Frees the persistent native resources exactly once. Each field is NULLed so the call is
   * idempotent (a double-free is a no-op) and a re-initialized conversation lazily rebuilds them on
   * next use. Must be called on the owning iterator thread before the state's arena is closed.
   */
  void free() {
    if (chain != null) {
      chain.free();
      chain = null;
    }
    if (draftBatch != null) {
      draftBatch.free();
      draftBatch = null;
    }
    if (verifyBatch != null) {
      verifyBatch.free();
      verifyBatch = null;
    }
  }

  /**
   * Greedy draft step with a confidence read, computed directly from the logit row with a two-pass
   * online softmax — no candidate-buffer fill, no native sampler, no sort. Returns the argmax token
   * id (the greedy pick, identical to a plain greedy sampler; strict {@code >} gives the lowest-id
   * tie-break) and writes its softmax probability {@code 1/Σexp(logit-maxLogit)} into
   * {@code probOut[0]}. Used only when adaptive early stop is enabled.
   */
  int draftGreedyConfident(
    MemorySegment logitsRow,
    int nVocab,
    float[] probOut
  ) {
    int argmax = 0;
    float maxLogit = logitsRow.getAtIndex(JAVA_FLOAT, 0);
    for (int id = 1; id < nVocab; id++) {
      float l = logitsRow.getAtIndex(JAVA_FLOAT, id);
      if (l > maxLogit) {
        maxLogit = l;
        argmax = id;
      }
    }
    double sum = 0.0;
    for (int id = 0; id < nVocab; id++) {
      sum += Math.exp(logitsRow.getAtIndex(JAVA_FLOAT, id) - maxLogit);
    }
    probOut[0] = (float) (1.0 / sum); // exp(maxLogit-maxLogit)=1, so the top prob is 1/sum
    return argmax;
  }

  Snapshot draft(LlamaSampler chain, MemorySegment logitsRow) {
    return snapshot(draftBuf, chain, logitsRow);
  }

  /** Selected token only — applies the chain to the target buffer and returns its choice. */
  int targetSelect(LlamaSampler chain, MemorySegment logitsRow) {
    return select(targetBuf, chain, logitsRow);
  }

  private static int select(
    LlamaTokenDataArray buf,
    LlamaSampler chain,
    MemorySegment logitsRow
  ) {
    buf.fill(logitsRow, 0);
    buf.apply(chain);
    return buf.selectedId();
  }

  private static Snapshot snapshot(
    LlamaTokenDataArray buf,
    LlamaSampler chain,
    MemorySegment logitsRow
  ) {
    buf.fill(logitsRow, 0);
    buf.apply(chain);
    int size = (int) buf.size();
    int[] ids = new int[size];
    float[] probs = new float[size];
    for (int i = 0; i < size; i++) {
      ids[i] = buf.idAt(i);
      probs[i] = buf.probabilityAt(i);
    }
    return new Snapshot(
      ids,
      probs,
      buf.selectedId(),
      buf.selectedProbability()
    );
  }

  /* ----- native-buffer verify ----- */

  /**
   * Rejection-sampling accept test operating directly on the persistent {@code targetBuf}: fills and
   * applies the chain once (leaving {@code targetBuf} populated for a possible {@link
   * #residualTargetScatter} / {@link #residualTargetPointMass}), then accepts the drafted token with
   * probability {@code min(1, p/qOfDrafted)} where {@code p} is its post-chain target probability.
   * {@code qOfDrafted} is the draft probability of the drafted token (the snapshot's selected
   * probability for model drafting, {@code 1} for the point-mass n-gram draft). RNG advances exactly
   * as the {@link Snapshot}-based {@link #accept} (one native apply + at most one accept coin), so the
   * sampled distribution is identical — this only avoids materializing the target snapshot.
   */
  boolean acceptTarget(
    LlamaSampler chain,
    MemorySegment logitsRow,
    int draftedToken,
    float qOfDrafted
  ) {
    targetBuf.fill(logitsRow, 0);
    targetBuf.apply(chain);
    if (qOfDrafted <= 0.0f) {
      return true;
    }
    double ratio = targetProbOf(draftedToken) / qOfDrafted;
    return ratio >= 1.0 || rng.nextDouble() < ratio;
  }

  /** Probability of {@code token} in the just-applied {@code targetBuf} (0 outside the kept support). */
  private float targetProbOf(int token) {
    int n = (int) targetBuf.size();
    for (int i = 0; i < n; i++) {
      if (targetBuf.idAt(i) == token) {
        return targetBuf.probabilityAt(i);
      }
    }
    return 0.0f;
  }

  /**
   * Residual draw over the already-applied {@code targetBuf} against a model-draft distribution:
   * samples from the normalized {@code (p - q)₊}, with {@code q} scattered by id. Allocation-free
   * (two passes; no {@code float[]} of differences). Must be called immediately after a rejecting
   * {@link #acceptTarget} for the same position (no intervening fill/apply).
   */
  int residualTargetScatter(Snapshot draft) {
    int[] draftIds = draft.ids();
    float[] draftProbs = draft.probs();
    for (int i = 0; i < draftIds.length; i++) {
      qScatter[draftIds[i]] = draftProbs[i];
    }
    try {
      int n = (int) targetBuf.size();
      double sum = 0.0;
      for (int i = 0; i < n; i++) {
        float diff = targetBuf.probabilityAt(i) - qScatter[targetBuf.idAt(i)];
        if (diff > 0.0f) {
          sum += diff;
        }
      }
      if (sum <= 0.0) {
        return targetBuf.selectedId();
      }
      double u = rng.nextDouble() * sum;
      double acc = 0.0;
      int last = targetBuf.selectedId();
      for (int i = 0; i < n; i++) {
        int id = targetBuf.idAt(i);
        float diff = targetBuf.probabilityAt(i) - qScatter[id];
        if (diff > 0.0f) {
          last = id;
          acc += diff;
          if (u < acc) {
            return id;
          }
        }
      }
      return last;
    } finally {
      for (int i = 0; i < draftIds.length; i++) {
        qScatter[draftIds[i]] = 0.0f;
      }
    }
  }

  /**
   * Residual draw over the already-applied {@code targetBuf} against a point-mass n-gram draft at
   * {@code token}: samples from the target probabilities with {@code token} removed (since
   * {@code (p - 1)₊ = 0} there) and renormalized. Allocation-free. Must be called immediately after
   * a rejecting {@link #acceptTarget} for the same position.
   */
  int residualTargetPointMass(int token) {
    int n = (int) targetBuf.size();
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
      if (targetBuf.idAt(i) != token) {
        float p = targetBuf.probabilityAt(i);
        if (p > 0.0f) {
          sum += p;
        }
      }
    }
    if (sum <= 0.0) {
      return targetBuf.selectedId();
    }
    double u = rng.nextDouble() * sum;
    double acc = 0.0;
    int last = targetBuf.selectedId();
    for (int i = 0; i < n; i++) {
      int id = targetBuf.idAt(i);
      if (id != token) {
        float p = targetBuf.probabilityAt(i);
        if (p > 0.0f) {
          last = id;
          acc += p;
          if (u < acc) {
            return id;
          }
        }
      }
    }
    return last;
  }
}
