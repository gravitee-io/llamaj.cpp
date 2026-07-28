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
package io.gravitee.llama.cpp.speculative;

import io.gravitee.llama.cpp.*;
import java.util.ArrayList;
import java.util.List;

/**
 * A speculative-decoding flavour's single-sequence round (BatchIterator drives the same draft
 * sources through its fused multi-sequence phases). One round = <b>draft</b> (flavour-specific) →
 * <b>verify</b> (one batched target decode) → <b>accept</b> (greedy longest-prefix, or rejection
 * sampling) → <b>rollback + emit</b>. Subclasses implement only what makes their flavour
 * different — how drafts are produced and how their side state advances:
 *
 * <ul>
 *   <li>{@link NgramSpeculativeDecoding} — proposals from the committed history; no draft KV.</li>
 *   <li>{@link ModelDraftSpeculativeDecoding} — a separate small model decoded token-by-token.</li>
 *   <li>{@link MtpSpeculativeDecoding} — the target's own nextn head.</li>
 *   <li>{@link Eagle3SpeculativeDecoding} — a trained EAGLE3 head over the target's captured
 *       layer inputs.</li>
 * </ul>
 *
 * The right variant is attached to the {@link ConversationState} by its {@code setDraft} /
 * {@code setNgram} / {@code setMtp} / {@code setEagle3} setter; the iterators just delegate.
 * The sampling math (draft snapshots, accept test, residual draws) lives in
 * {@link Speculation}; detokenization/emission stays on the iterator
 * ({@code emitSpeculative}). Greedy configs are lossless w.r.t. plain greedy decoding; sampling
 * configs are exact samplers of the target distribution. Implementations are stateless
 * singletons — all per-conversation state lives on the {@link ConversationState}.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public abstract sealed class SpeculativeDecoding
  permits
    ModelDraftSpeculativeDecoding,
    NgramSpeculativeDecoding,
    HiddenStateSpeculativeDecoding {

  /**
   * Brings the flavour's draft-side state in line with the freshly-processed prompt (called
   * once after {@code processPrompt}). Default: nothing to prefill.
   */
  public void prefill(ConversationState state) {}

  /**
   * Runs one draft → verify → accept round and returns the committed tokens as outputs.
   * Requires {@code state.getNewTokenId()} to be set (the last token, not yet in any KV);
   * updates the state's {@code nPast}/{@code newTokenId} and sets the finish reason on EOG or
   * token-limit. On failure the state's persistent native scratch is released (idempotent).
   */
  public final List<LlamaOutput> round(
    LlamaIterator<?> it,
    ConversationState state
  ) {
    try {
      return roundImpl(it, state);
    } catch (RuntimeException e) {
      state.freeSpeculativeScratch();
      throw e;
    }
  }

  /** The flavour-specific round body. */
  protected abstract List<LlamaOutput> roundImpl(
    LlamaIterator<?> it,
    ConversationState state
  );

  /* -------------------------------- shared mechanics -------------------------------- */

  /** Accepted-prefix length + the correction (partial accept) or bonus (full accept) token. */
  record Verdict(int matched, int extra) {}

  /** One batched target decode of {@code [idLast, drafted[0..m-1]]} with logits on every row. */
  static void decodeVerify(
    LlamaBatch verifyBatch,
    LlamaContext target,
    int idLast,
    int nPast,
    int[] drafted,
    int m,
    List<Integer> seq
  ) {
    verifyBatch.clear();
    verifyBatch.add(idLast, nPast, seq, true);
    for (int i = 0; i < m; i++) {
      verifyBatch.add(drafted[i], nPast + 1 + i, seq, true);
    }
    if (verifyBatch.decode(target) != 0) {
      throw new LlamaException("Speculative verify decode failed");
    }
  }

  /**
   * Applies the budget EOG ramp to verify row {@code i}, which is the distribution after {@code i}
   * more tokens. Biasing before the row is read keeps the acceptance test and the residual draw on
   * the same target, so the round stays an exact sampler of the biased distribution.
   */
  private static void biasVerifyRow(
    LlamaIterator<?> it,
    ConversationState state,
    LlamaContext target,
    int i
  ) {
    if (!state.hasEogRamp()) {
      return;
    }
    var row = it.logitsRow(target, i, target.nVocab());
    if (it.biasEogRow(state, row, state.getAnswerTokens() + i)) {
      state.setEogRampApplied(true);
    }
  }

  /**
   * Accepts drafts against the verify rows: greedy longest-prefix when the config is greedy
   * (lossless), otherwise rejection sampling ({@code min(1, p/q)} accept + residual draw on the
   * first rejection — an exact sampler of the target distribution). {@code snaps == null} means
   * a point-mass draft (n-gram, q = 1); otherwise q comes from the draft snapshots.
   */
  static Verdict accept(
    LlamaIterator<?> it,
    ConversationState state,
    Speculation spec,
    LlamaContext target,
    int[] drafted,
    Speculation.Snapshot[] snaps,
    int m
  ) {
    LlamaSampler chain = spec.chain();
    if (spec.isGreedy()) {
      int matched = 0;
      int correction = -1;
      for (int i = 0; i < m; i++) {
        biasVerifyRow(it, state, target, i);
        int t = chain.sample(target, i);
        if (t == drafted[i]) {
          matched++;
        } else {
          correction = t;
          break;
        }
      }
      if (matched == m) {
        biasVerifyRow(it, state, target, m);
      }
      int extra = matched == m ? chain.sample(target, m) : correction;
      return new Verdict(matched, extra);
    }

    int nVocab = target.nVocab();
    int matched = 0;
    int extra = -1;
    for (int i = 0; i < m; i++) {
      biasVerifyRow(it, state, target, i);
      float q = snaps == null ? 1.0f : snaps[i].selectedProbability();
      if (
        spec.acceptTarget(chain, it.logitsRow(target, i, nVocab), drafted[i], q)
      ) {
        matched++;
      } else {
        extra = snaps == null
          ? spec.residualTargetPointMass(drafted[i])
          : spec.residualTargetScatter(snaps[i]);
        break;
      }
    }
    if (matched == m) {
      biasVerifyRow(it, state, target, m);
      extra = spec.targetSelect(chain, it.logitsRow(target, m, nVocab));
    }
    return new Verdict(matched, extra);
  }

  /** Emits the accepted drafts then the extra token (stops early on EOG/quota). */
  static List<LlamaOutput> emitCommitted(
    LlamaIterator<?> it,
    ConversationState state,
    int[] drafted,
    Verdict v
  ) {
    List<LlamaOutput> out = new ArrayList<>();
    boolean cont = true;
    for (int i = 0; i < v.matched() && cont; i++) {
      cont = it.emitSpeculative(state, drafted[i], out);
    }
    if (cont) {
      it.emitSpeculative(state, v.extra(), out);
    }
    return out;
  }

  /**
   * Advances the conversation to the accepted boundary and records accept statistics. Also
   * updates the committed-token history to the tokens whose KV rows survived the rollback:
   * {@code idLast} (the pre-round newTokenId, decoded at the old nPast) plus the accepted
   * drafts — keeping {@code history.size() == nPast}.
   */
  static void commit(
    ConversationState state,
    Verdict v,
    int[] drafted,
    int nDrafted,
    int newNPast
  ) {
    var history = state.getTokenHistory();
    history.truncate(state.getNPast()); // mirror the KV rollback (defensive no-op normally)
    history.append(state.getNewTokenId()); // idLast — setNewTokenId(extra) happens below
    for (int i = 0; i < v.matched(); i++) {
      history.append(drafted[i]);
    }
    state.setNPast(newNPast);
    state.setNewTokenId(v.extra());
    state.recordSpeculation(nDrafted, v.matched());
  }
}
