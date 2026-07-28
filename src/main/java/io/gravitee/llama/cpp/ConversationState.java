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

import static io.gravitee.llama.cpp.GenerationState.ANSWER;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import io.gravitee.llama.cpp.LlamaTokenizer.TokenizerResponse;
import io.gravitee.llama.cpp.draft.Eagle3Draft;
import io.gravitee.llama.cpp.draft.MtpDraft;
import io.gravitee.llama.cpp.draft.NgramIndex;
import io.gravitee.llama.cpp.modules.PromptMemory;
import io.gravitee.llama.cpp.modules.StateEvaluation;
import io.gravitee.llama.cpp.modules.StopString;
import io.gravitee.llama.cpp.modules.TokenHistory;
import io.gravitee.llama.cpp.modules.TokenTracking;
import io.gravitee.llama.cpp.speculative.*;
import io.gravitee.llama.cpp.utils.Utf8Decoder;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class ConversationState {

  // Resources (owned by this state)
  private final Arena arena;
  private final LlamaContext context;
  private final LlamaTokenizer tokenizer;
  private final LlamaSampler sampler;

  // Identity & position
  private final int sequenceId;
  private int nPast = 0;
  private String promptText;

  // KV prefix reuse: how many leading prompt tokens' KV rows are reused from a previous
  // generation on this sequence (processPrompt then only decodes the suffix), and whether the
  // sequence's KV should be retained (not wiped) when this state finishes / is cleaned up.
  private int reusePrefixTokens = 0;
  private boolean retainKv = false;
  // Whether the requested prefix reuse was actually honored by the memory backend. Set to
  // false when processPrompt's partial seq_rm is rejected (recurrent/hybrid models whose
  // rollback-snapshot window n_rs_seq cannot cover the required rewind) and a cold full
  // prefill is performed instead.
  private boolean prefixReuseHonored = true;

  // Committed-token history: token ids whose KV rows are resident, positions [0, nPast).
  // history.size() == nPast at all stable points (see TokenHistory).
  private final TokenHistory tokenHistory = new TokenHistory();

  // Tokenization
  private TokenizerResponse tokenized;

  // Tracking & state
  private final TokenTracking tokenTracking = new TokenTracking();
  private final PromptMemory promptMemory = new PromptMemory();
  private final StopString stopString = new StopString();
  private final StateEvaluation stateEvaluation = new StateEvaluation();
  private final Utf8Decoder decoder = new Utf8Decoder();

  // Generation state
  private GenerationState generationState = ANSWER;
  private FinishReason finishReason;
  private boolean finished;

  // Configuration
  private int maxTokens = -1;

  // Budget-aware EOG bias ("soft landing"): as the token budget runs down, end-of-generation
  // logits get an increasing boost so the model closes its sentence instead of being severed
  // mid-word. Disabled by default (startFraction < 0) — enabling it changes the sampled
  // distribution, so the default path stays bit-identical.
  private float eogRampStart = -1f;
  private char lastNonSpaceChar = 0;
  private boolean lastEndsLine = false;
  private float eogRampMaxBias = 0f;
  // Set on the step whose logits were biased, so an EOG produced under pressure is still
  // reported as LENGTH rather than a natural STOP.
  private boolean eogRampApplied = false;
  private int topLogprobs = 0;

  // Optional speculative decoding: a draft context (separate small model, same vocab) whose
  // KV is kept in lockstep with this state's target context. When set, the iterators run a
  // draft→verify→accept cycle (greedy or rejection-sampling) instead of single-token decoding.
  // For n-gram (prompt-lookup) drafting the draftContext is null and proposals come from `history`.
  private LlamaContext draftContext;
  private SpeculativeConfig speculativeConfig;
  private Speculation speculation;
  private long nDrafted;
  private long nAccepted;
  // Rolling accept rate driving the adaptive per-round draft budget (getNDraft). Starts
  // optimistic so the first rounds draft at the configured maximum.
  private static final double EWMA_ALPHA = 0.3;
  private double ewmaAcceptRate = 1.0;
  // Deferred draft-KV fill from a full-accept round (model-draft flavour); -1 = none.
  private int pendingDraftFillToken = -1;
  private int pendingDraftFillPos;

  // N-gram (prompt-lookup) drafting history + position index (the committed token stream
  // prompt+generated, on the heap, NOT the confined arena). Built lazily by setNgram().
  private NgramIndex ngramIndex;

  // MTP self-speculation (setMtp) / EAGLE3 head drafting (setEagle3) proposal state.
  private MtpDraft mtpDraft;
  private Eagle3Draft eagle3Draft;

  // The speculative flavour attached by setDraft/setNgram/setMtp/setEagle3 (stateless singleton).
  private SpeculativeDecoding speculativeDecoding;
  private final List<StateBounds> stateBounds = new ArrayList<>();
  private List<MtmdMedia> media = new ArrayList<>();

  // Iteration state (used by iterator)
  Integer newTokenId;
  String piece;
  // Number of generated tokens the current `piece` covers: 1 normally, 0 while a multi-token
  // marker prefix is buffered (empty piece), N when a buffered marker resolves.
  int pieceTokens = 1;
  Logprobs logprobs;

  private ConversationState(
    Arena arena,
    LlamaContext context,
    LlamaTokenizer tokenizer,
    LlamaSampler sampler,
    int sequenceId
  ) {
    this.arena = arena;
    this.context = context;
    this.tokenizer = tokenizer;
    this.sampler = sampler;
    this.sequenceId = sequenceId;
  }

  /**
   * Creates a new conversation state with resources and sequence ID.
   *
   * @param arena The memory arena
   * @param context The LlamaContext to use
   * @param tokenizer The tokenizer to use
   * @param sampler The sampler to use
   * @param sequenceId The sequence ID for this conversation
   * @return A new conversation state
   */
  public static ConversationState create(
    Arena arena,
    LlamaContext context,
    LlamaTokenizer tokenizer,
    LlamaSampler sampler,
    int sequenceId
  ) {
    return new ConversationState(
      arena,
      context,
      tokenizer,
      sampler,
      sequenceId
    );
  }

  /**
   * Creates a new conversation state with default sequence ID (0).
   */
  public static ConversationState create(
    Arena arena,
    LlamaContext context,
    LlamaTokenizer tokenizer,
    LlamaSampler sampler
  ) {
    return new ConversationState(arena, context, tokenizer, sampler, 0);
  }

  /**
   * Gets the arena used by this conversation.
   */
  public Arena getArena() {
    return arena;
  }

  /**
   * Initializes this conversation with a prompt.
   * Note: This method resets all generation state including media.
   * Call {@link #setMedia(List)} after this method if multimodal input is needed.
   *
   * @param prompt The prompt text
   * @return This state for chaining
   */
  public ConversationState initialize(String prompt) {
    return initialize(prompt, 0);
  }

  /**
   * Initializes this conversation with a prompt, reusing the KV rows of the first
   * {@code reusePrefixTokens} prompt tokens already resident for this sequence (from a previous
   * generation retained via {@link #setRetainKv(boolean)} / {@code removeState(id, true)}).
   * {@code processPrompt} then wipes the sequence's KV from {@code reusePrefixTokens} on and
   * decodes only the prompt suffix at absolute positions.
   *
   * <p>{@code reusePrefixTokens} must be within {@code [0, tokenized.size()]}; a value equal to
   * the full prompt length is clamped to {@code size - 1}: the last prompt token must be
   * re-decoded to produce logits — its KV row gets trimmed and rewritten identically.
   *
   * <p>The committed-token history restarts as the FULL new tokenized prompt (all of it becomes
   * KV-resident once prefill completes).
   *
   * @param prompt            The prompt text
   * @param reusePrefixTokens Number of leading prompt tokens whose KV rows are reused
   * @return This state for chaining
   */
  public ConversationState initialize(String prompt, int reusePrefixTokens) {
    this.tokenized = tokenizer.tokenize(arena, prompt);
    int size = tokenized.size();
    if (reusePrefixTokens < 0 || reusePrefixTokens > size) {
      throw new LlamaException(
        "reusePrefixTokens (" +
          reusePrefixTokens +
          ") must be within [0, " +
          size +
          "] (the tokenized prompt length)"
      );
    }
    if (reusePrefixTokens == size && size > 0) {
      // Clamp: the last prompt token is always re-decoded to produce logits.
      reusePrefixTokens = size - 1;
    }
    this.reusePrefixTokens = reusePrefixTokens;
    this.prefixReuseHonored = true;
    int[] promptTokens = new int[size];
    for (int i = 0; i < size; i++) {
      promptTokens[i] = tokenized.data().getAtIndex(JAVA_INT, i);
    }
    this.tokenHistory.initialize(promptTokens);
    this.promptText = prompt;
    this.tokenTracking.initialize(tokenized.size());
    this.stateEvaluation.initialize(new StateEvaluation.Config(stateBounds));
    this.generationState = stateEvaluation.initialState(prompt);
    this.finishReason = null;
    this.newTokenId = null;
    this.piece = null;
    this.pieceTokens = 1;
    this.logprobs = null;
    this.nPast = 0;
    this.decoder.reset();
    this.media.clear();
    return this;
  }

  public String getPromptText() {
    return promptText;
  }

  /**
   * Sets the maximum number of tokens to generate.
   */
  public ConversationState setMaxTokens(int maxTokens) {
    this.maxTokens = maxTokens;
    return this;
  }

  /**
   * Enables speculative decoding: {@code draftContext} (a small model sharing this target's
   * tokenizer/vocab) proposes {@code config.nDraft()} tokens per step that the target verifies
   * in a single decode. Greedy config is lossless w.r.t. greedy decoding; a sampling config
   * (temp/top-k/top-p) uses rejection sampling, preserving the target distribution. The state's
   * main sampler is bypassed for accepted tokens — speculative sampling is governed by
   * {@code config}.
   *
   * @param draftContext A context over the draft model (must share the target's vocab size)
   * @param config       Speculative decoding configuration
   */
  public ConversationState setDraft(
    LlamaContext draftContext,
    SpeculativeConfig config
  ) {
    if (draftContext.nVocab() != context.nVocab()) {
      throw new LlamaException(
        "Draft vocab size (" +
          draftContext.nVocab() +
          ") differs from target (" +
          context.nVocab() +
          ") — speculative decoding requires a shared tokenizer/vocab"
      );
    }
    this.draftContext = draftContext;
    this.speculativeConfig = config;
    this.speculation = new Speculation(arena, context.nVocab(), config);
    this.speculativeDecoding = ModelDraftSpeculativeDecoding.INSTANCE;
    return this;
  }

  /**
   * Enables n-gram (prompt-lookup) speculative decoding: proposes up to {@code config.nDraft()}
   * tokens per round by matching the last {@code config.ngram()} committed tokens against this
   * conversation's generation history — no draft model, no draft forward pass. The target still
   * verifies every proposed token, so output is lossless (greedy) / exact (sampling). Requires an
   * n-gram config ({@code config.isNgram()}).
   */
  public ConversationState setNgram(SpeculativeConfig config) {
    if (!config.isNgram()) {
      throw new LlamaException(
        "setNgram requires an n-gram config (ngram >= 1); use setDraft for model drafting"
      );
    }
    this.speculativeConfig = config;
    this.speculation = new Speculation(arena, context.nVocab(), config);
    this.ngramIndex = new NgramIndex(config.ngram());
    this.speculativeDecoding = NgramSpeculativeDecoding.INSTANCE;
    return this;
  }

  /**
   * Enables MTP (nextn) self-speculative decoding: the target model's own multi-token-prediction
   * head proposes tokens — no separate draft model. Requires a model with {@code n_layer_nextn > 0},
   * the staging nextn API in the loaded llama.cpp
   * ({@link LlamaExt#available()}), and {@code mtpContext} built from the <b>same model</b> as this
   * state's target with {@code LlamaContextParams.ctxTypeMtp().ctxOther(target).nRsSeq(>0)}.
   *
   * <p>Enables embeddings output on the target context (the MTP head is seeded with the target's
   * post-norm hidden states). Greedy config is lossless; sampling configs use rejection sampling.
   * N-gram configs are rejected.
   *
   * @param mtpContext An MTP draft context over the target's model (caller-owned, like a draft ctx)
   * @param config     Speculative decoding configuration ({@code ngram} must be 0)
   */
  public ConversationState setMtp(
    LlamaContext mtpContext,
    SpeculativeConfig config
  ) {
    if (config.isNgram()) {
      throw new LlamaException(
        "setMtp requires a model-draft config (ngram == 0)"
      );
    }
    if (!LlamaExt.available()) {
      throw new LlamaException(
        "MTP requires the staging nextn API in the loaded libllama:\n" +
          LlamaExt.resolutionReport()
      );
    }
    // The MTP context must share the target's vocab (it is the same model).
    if (mtpContext.nVocab() != context.nVocab()) {
      throw new LlamaException(
        "MTP context vocab (" +
          mtpContext.nVocab() +
          ") differs from target (" +
          context.nVocab() +
          ") — the MTP context must be built from the target's model"
      );
    }
    // Seed extraction reads the target's post-norm hidden rows.
    LlamaRuntime.llama_set_embeddings(context.segment, true);
    // The MTP graph stores its own nextn hidden per output row for nDraft>1 chaining.
    LlamaExt.setEmbeddingsNextn(mtpContext, true, false);
    this.speculativeConfig = config;
    this.speculation = new Speculation(arena, context.nVocab(), config);
    this.mtpDraft = new MtpDraft(
      arena,
      mtpContext,
      context.getModel().nEmbdOut()
    );
    this.speculativeDecoding = MtpSpeculativeDecoding.INSTANCE;
    return this;
  }

  /**
   * Enables EAGLE3 speculative decoding: a separate tiny trained head model (GGUF arch
   * {@code eagle3}) proposes tokens from the target's intermediate hidden states. Works with
   * targets that carry no nextn head (dense models). Requires the staging layer-input API
   * ({@link LlamaExt#eagle3Available()}), a head model declaring exactly 3 target extract layers,
   * and a shared vocab.
   *
   * <p>Enables capture of the declared target layers' input hiddens on the target context and
   * nextn (pre-norm) capture on the head context. Greedy config is lossless; sampling configs use
   * rejection sampling. N-gram configs are rejected.
   *
   * @param eagle3Context A context over the EAGLE3 head model (caller-owned)
   * @param eagle3Model   The EAGLE3 head model (for its target-layer ids and hidden size)
   * @param config        Speculative decoding configuration ({@code ngram} must be 0)
   */
  public ConversationState setEagle3(
    LlamaContext eagle3Context,
    LlamaModel eagle3Model,
    SpeculativeConfig config
  ) {
    if (config.isNgram()) {
      throw new LlamaException(
        "setEagle3 requires a model-draft config (ngram == 0)"
      );
    }
    if (!LlamaExt.eagle3Available()) {
      throw new LlamaException(
        "EAGLE3 requires the staging layer-input API in the loaded libllama:\n" +
          LlamaExt.eagle3ResolutionReport()
      );
    }
    int[] layerIds = LlamaExt.targetLayerIds(eagle3Model);
    if (layerIds.length != 3) {
      throw new LlamaException(
        "Not an EAGLE3 head model: expected 3 target extract layers, got " +
          layerIds.length
      );
    }
    if (eagle3Context.nVocab() != context.nVocab()) {
      throw new LlamaException(
        "EAGLE3 head vocab (" +
          eagle3Context.nVocab() +
          ") differs from target (" +
          context.nVocab() +
          ") — the head must be converted against this target model"
      );
    }
    // Capture the declared target layers' input hiddens on every target decode.
    for (int id : layerIds) {
      LlamaExt.setEmbeddingsLayerInp(context, id, true);
    }
    // Head pre-norm hidden capture: encoder g_embd rows + decoder chain seeds.
    LlamaExt.setEmbeddingsNextn(eagle3Context, true, true);
    this.speculativeConfig = config;
    this.speculation = new Speculation(arena, context.nVocab(), config);
    this.eagle3Draft = new Eagle3Draft(
      arena,
      eagle3Context,
      layerIds,
      context.getModel().nEmbdOut(),
      eagle3Model.nEmbdOut(),
      config.nDraft()
    );
    this.speculativeDecoding = Eagle3SpeculativeDecoding.INSTANCE;
    return this;
  }

  public boolean hasDraft() {
    return draftContext != null;
  }

  /** Whether any speculative drafting (model or n-gram) is enabled on this state. */
  public boolean isSpeculative() {
    return speculation != null;
  }

  /** Whether n-gram (prompt-lookup) drafting is enabled (vs model drafting). */
  public boolean isNgram() {
    return speculativeConfig != null && speculativeConfig.isNgram();
  }

  /** Whether MTP (nextn) self-speculation is enabled. */
  public boolean isMtp() {
    return mtpDraft != null;
  }

  /** Whether EAGLE3 head drafting is enabled. */
  public boolean isEagle3() {
    return eagle3Draft != null;
  }

  public MtpDraft getMtpDraft() {
    return mtpDraft;
  }

  public Eagle3Draft getEagle3Draft() {
    return eagle3Draft;
  }

  /** Frees all speculative persistent native scratch (idempotent; safe when none is set). */
  public void freeSpeculativeScratch() {
    if (speculation != null) {
      speculation.free();
    }
    if (mtpDraft != null) {
      mtpDraft.free();
    }
    if (eagle3Draft != null) {
      eagle3Draft.free();
    }
  }

  public LlamaContext getDraftContext() {
    return draftContext;
  }

  /**
   * Tokens to draft this round. Fixed configs always return the configured {@code nDraft}.
   * Adaptive configs ({@code pMin > 0}) additionally scale the per-round budget with the
   * recent accept rate (EWMA): a draft that keeps getting rejected wastes its decodes past
   * the first position, so the budget shrinks toward {@code draftMin} and recovers as
   * accepts return. Verify-batch capacity is sized off the configured maximum, so the
   * dynamic value is always safe.
   */
  public int getNDraft() {
    int max = speculativeConfig.nDraft();
    if (!speculativeConfig.isAdaptive()) {
      return max;
    }
    int min = speculativeConfig.draftMin();
    int k = (int) Math.round(min + ewmaAcceptRate * (max - min));
    return Math.max(min, Math.min(max, k));
  }

  public Speculation getSpeculation() {
    return speculation;
  }

  /** The speculative flavour enabled on this state (null when not speculative). */
  public SpeculativeDecoding getSpeculativeDecoding() {
    return speculativeDecoding;
  }

  /**
   * Seeds the n-gram history with the prompt tokens plus the first sampled token (idLast). Must be
   * called once after {@code processPrompt} has set {@code newTokenId} (n-gram mode only).
   */
  void seedNgramHistory() {
    ngramIndex.clear();
    int promptLen = tokenized.size();
    var data = tokenized.data();
    for (int i = 0; i < promptLen; i++) {
      ngramIndex.append(data.getAtIndex(JAVA_INT, i));
    }
    ngramIndex.append(newTokenId); // idLast, position == nPast (not yet in KV)
  }

  /** Appends one committed token to the n-gram history (and its index). */
  public void appendHistory(int token) {
    ngramIndex.append(token);
  }

  /**
   * Proposes up to {@code kMax} draft tokens by finding the most recent earlier occurrence of the
   * last {@code ngram} history tokens and returning the tokens that followed it (via the position
   * index). Returns an empty array when there is no match (the round then degenerates to a single
   * target decode). Pure heap/CPU work — no native calls, no draft KV. A wrong proposal only lowers
   * the accept rate; the target verify is the sole arbiter of emitted tokens.
   */
  public int[] proposeNgram(int kMax) {
    return ngramIndex.propose(kMax);
  }

  /** Accumulates speculative accept statistics for {@link #acceptRate()} and {@link #getNDraft()}. */
  public void recordSpeculation(int drafted, int accepted) {
    this.nDrafted += drafted;
    this.nAccepted += accepted;
    if (drafted > 0) {
      ewmaAcceptRate =
        (1 - EWMA_ALPHA) * ewmaAcceptRate +
        EWMA_ALPHA * ((double) accepted / drafted);
    }
  }

  /* ----- deferred draft-KV fill (model-draft flavour, see ModelDraftSpeculativeDecoding) ----- */

  public boolean hasPendingDraftFill() {
    return pendingDraftFillToken != -1;
  }

  public int pendingDraftFillToken() {
    return pendingDraftFillToken;
  }

  public int pendingDraftFillPos() {
    return pendingDraftFillPos;
  }

  public void setPendingDraftFill(int token, int pos) {
    this.pendingDraftFillToken = token;
    this.pendingDraftFillPos = pos;
  }

  public void clearPendingDraftFill() {
    this.pendingDraftFillToken = -1;
  }

  /** Fraction of drafted tokens accepted so far — a sanity check on the speedup. */
  public double acceptRate() {
    return nDrafted == 0 ? 0.0 : (double) nAccepted / nDrafted;
  }

  /**
   * Enables log-probability collection for each generated token.
   *
   * <p>When set to a value greater than zero, each {@link LlamaOutput} returned by the
   * iterator will contain a {@link Logprobs} object with the sampled token's log-probability
   * and the {@code topLogprobs} most-likely alternatives at that position.
   *
   * <p>Setting this to {@code 0} (the default) disables logprobs collection entirely,
   * which avoids the overhead of reading and sorting the full vocabulary logit vector.
   *
   * @param topLogprobs Number of top-alternative tokens to include (0 = disabled,
   *                    max 20 as per OpenAI convention)
   * @return This state for chaining
   */
  public ConversationState setTopLogprobs(int topLogprobs) {
    this.topLogprobs = topLogprobs;
    return this;
  }

  public int getTopLogprobs() {
    return topLogprobs;
  }

  /**
   * Sets stop strings for this conversation.
   */
  public ConversationState setStopStrings(List<String> stopStrings) {
    this.stopString.initialize(stopStrings);
    int maxStringSize = stopStrings
      .stream()
      .mapToInt(String::length)
      .max()
      .orElse(0);
    this.promptMemory.initialize(maxStringSize);
    return this;
  }

  /**
   * Configures reasoning token detection.
   */
  public ConversationState setReasoning(String tokenStart, String tokenEnd) {
    this.stateBounds.add(
      new StateBounds(GenerationState.REASONING, tokenStart, tokenEnd)
    );
    return this;
  }

  /**
   * Configures tool call detection.
   */
  public ConversationState setToolCall(String tokenStart, String tokenEnd) {
    this.stateBounds.add(
      new StateBounds(GenerationState.TOOLS, tokenStart, tokenEnd)
    );
    return this;
  }

  /**
   * Configures tool call detection where the channel can be opened more than one way — Harmony
   * uses both {@code commentary} and {@code analysis}. Each alternative must be complete from the
   * start of a run; with only one configured, the other leaks into reasoning as raw text.
   */
  public ConversationState setToolCall(
    java.util.List<String> tokenStarts,
    String tokenEnd
  ) {
    this.stateBounds.add(
      new StateBounds(GenerationState.TOOLS, tokenStarts, tokenEnd)
    );
    return this;
  }

  public List<MtmdMedia> getMedia() {
    return media;
  }

  public ConversationState setMedia(List<MtmdMedia> media) {
    this.media = media;
    return this;
  }

  /**
   * @deprecated Use {@link #getMedia()} instead. Kept for backward compatibility.
   */
  @Deprecated
  public List<MtmdImage> getImages() {
    return media
      .stream()
      .filter(m -> m instanceof MtmdImage)
      .map(m -> (MtmdImage) m)
      .toList();
  }

  /**
   * @deprecated Use {@link #setMedia(List)} instead. Kept for backward compatibility.
   */
  @Deprecated
  public ConversationState setImages(List<MtmdImage> images) {
    this.media = new ArrayList<>(images);
    return this;
  }

  // Resource getters
  public LlamaContext getContext() {
    return context;
  }

  public LlamaTokenizer getTokenizer() {
    return tokenizer;
  }

  public LlamaSampler getSampler() {
    return sampler;
  }

  public Utf8Decoder getDecoder() {
    return decoder;
  }

  // State getters
  public int getSequenceId() {
    return sequenceId;
  }

  public int getNPast() {
    return nPast;
  }

  public void setNPast(int nPast) {
    this.nPast = nPast;
  }

  public void incrementNPast() {
    this.nPast++;
  }

  /** Leading prompt tokens whose KV rows are reused by this initialization (see initialize). */
  public int getReusePrefixTokens() {
    return reusePrefixTokens;
  }

  /**
   * Resets the reuse offset to 0. Called by the prefill when the memory backend rejects a
   * partial trim (recurrent/hybrid models) and a cold full prefill is performed instead, so
   * observers see the reuse that actually happened.
   */
  public void clearReusePrefixTokens() {
    this.reusePrefixTokens = 0;
    this.prefixReuseHonored = false;
  }

  /**
   * Whether the prefix reuse requested via {@link #initialize(String, int)} was actually honored
   * by the prefill. {@code false} means the memory backend rejected the partial trim — attention
   * models never reject it, but recurrent/hybrid models (SSM, gated-deltanet: e.g. qwen3next,
   * qwen3.5/3.6) can only rewind their recurrent state within the per-token snapshot window
   * ({@code n_rs_seq}); a rewind farther back than that forces a cold full prefill and this
   * returns {@code false}. Callers advertising prefix reuse (cross-request KV caches) should
   * check this after prompt processing rather than trusting the requested reuse count.
   */
  public boolean isPrefixReuseHonored() {
    return prefixReuseHonored;
  }

  /**
   * When {@code true}, this sequence's KV cache is retained (not wiped) when the state finishes
   * naturally or is cleaned up by its iterator — enabling a later
   * {@code initialize(prompt, reusePrefixTokens)} on the same sequence id to reuse the resident
   * prefix. Explicit {@code BatchIterator.removeState(id, false)}, {@code stop()} and
   * decode-error teardown always wipe regardless.
   */
  public ConversationState setRetainKv(boolean retainKv) {
    this.retainKv = retainKv;
    return this;
  }

  public boolean isRetainKv() {
    return retainKv;
  }

  /**
   * Snapshot of the committed token ids — the tokens whose KV rows are resident for this
   * sequence, positions {@code [0, nPast)}. Length {@code == nPast} at all stable points
   * (text path; the multimodal path does not maintain the history).
   */
  public int[] committedTokens() {
    return tokenHistory.toArray();
  }

  /** Internal: the committed-token history backing {@link #committedTokens()}. */
  public TokenHistory getTokenHistory() {
    return tokenHistory;
  }

  public TokenizerResponse getTokenized() {
    return tokenized;
  }

  public TokenTracking getTokenTracking() {
    return tokenTracking;
  }

  public PromptMemory getPromptMemory() {
    return promptMemory;
  }

  public StopString getStopString() {
    return stopString;
  }

  public StateEvaluation getStateEvaluation() {
    return stateEvaluation;
  }

  public GenerationState getGenerationState() {
    return generationState;
  }

  public void setGenerationState(GenerationState generationState) {
    this.generationState = generationState;
  }

  public FinishReason getFinishReason() {
    return finishReason;
  }

  public void setFinishReason(FinishReason finishReason) {
    this.finishReason = finishReason;
  }

  /**
   * Returns {@code true} when the model has actually stopped generating tokens
   * (EOG token or token limit). Distinct from {@link #getFinishReason()} which
   * may be set as a marker (e.g. {@code TOOL_CALL}) while generation continues.
   */
  public boolean isFinished() {
    return finished;
  }

  /**
   * Marks the model as done generating. Set by {@code shouldContinue()} when
   * EOG or LENGTH is detected.
   */
  public void setFinished(boolean finished) {
    this.finished = finished;
  }

  public int getMaxTokens() {
    return maxTokens;
  }

  /**
   * Turns {@link #setMaxTokens(int) maxTokens} from a hard cut into a soft landing: past
   * {@code startFraction} of the budget, every end-of-generation token's logit is raised by a
   * quadratic ramp reaching {@code maxBias} nats at the cap, so the model is increasingly likely
   * to finish its sentence on its own. The hard cap remains as a backstop.
   *
   * <p>Costs: completions typically end <em>short</em> of the budget, so {@code maxTokens} becomes
   * a target rather than a ceiling; and the sampled distribution is no longer the model's own, so
   * output is not comparable with an unbiased run at the same seed. An EOG produced while the ramp
   * is active is still reported as {@link FinishReason#LENGTH} — the budget caused it, and callers
   * (agent loops, OpenAI-compatible clients) rely on that to know the answer was cut short.
   *
   * <p>The boost only applies where the text can actually stop — after sentence-terminating
   * punctuation or a line break, widened to clause punctuation in the second half of the ramp.
   * Biasing every step just stops mid-clause a few tokens early, which is the same severed
   * output arriving sooner; the gate is what turns the budget into a landing.
   *
   * <p>Applies under speculative decoding too: rejection sampling is exact for any target
   * distribution as long as the acceptance test and the residual draw see the same one.
   *
   * @param startFraction where the ramp begins, as a fraction of maxTokens (e.g. 0.75); negative
   *                      disables the ramp
   * @param maxBias       logit boost in nats at the cap. EOG sits far below the running text in
   *                      raw logits mid-answer — measured, 24 never wins and 100 does, so size
   *                      this generously rather than by intuition about probabilities
   */
  public ConversationState setEogRamp(float startFraction, float maxBias) {
    this.eogRampStart = startFraction;
    this.eogRampMaxBias = maxBias;
    return this;
  }

  /** Whether a budget-aware EOG ramp is configured (see {@link #setEogRamp(float, float)}). */
  public boolean hasEogRamp() {
    return eogRampStart >= 0f && maxTokens > 0;
  }

  public float getEogRampStart() {
    return eogRampStart;
  }

  public float getEogRampMaxBias() {
    return eogRampMaxBias;
  }

  /** Marks that this step's logits were biased, so an EOG here is budget-driven, not natural. */
  public void setEogRampApplied(boolean applied) {
    this.eogRampApplied = applied;
  }

  public boolean isEogRampApplied() {
    return eogRampApplied;
  }

  public Integer getNewTokenId() {
    return newTokenId;
  }

  public void setNewTokenId(Integer newTokenId) {
    this.newTokenId = newTokenId;
  }

  public String getPiece() {
    return piece;
  }

  public void setPiece(String piece) {
    this.piece = piece;
    // Remember where the text currently stands, for the EOG ramp's boundary gate. Empty pieces
    // (a buffered marker prefix) must not clear it: nothing was emitted, so the last real
    // boundary still holds.
    if (piece != null && !piece.isEmpty()) {
      lastEndsLine = piece.charAt(piece.length() - 1) == '\n';
      for (int i = piece.length() - 1; i >= 0; i--) {
        char c = piece.charAt(i);
        if (!Character.isWhitespace(c)) {
          lastNonSpaceChar = c;
          break;
        }
      }
    }
  }

  /** Last non-whitespace character emitted, or {@code 0} before any output. */
  public char getLastNonSpaceChar() {
    return lastNonSpaceChar;
  }

  /** Whether the last emitted piece ended a line. */
  public boolean isLastEndsLine() {
    return lastEndsLine;
  }

  /** Number of generated tokens covered by the current {@link #getPiece() piece}. */
  public int getPieceTokens() {
    return pieceTokens;
  }

  public void setPieceTokens(int pieceTokens) {
    this.pieceTokens = pieceTokens;
  }

  public Logprobs getLogprobs() {
    return logprobs;
  }

  public void setLogprobs(Logprobs logprobs) {
    this.logprobs = logprobs;
  }

  // Token count accessors
  public int getInputTokens() {
    return tokenTracking.getInputTokenCount();
  }

  public int getAnswerTokens() {
    return tokenTracking.getOutputTokenCount(ANSWER);
  }

  public int getReasoningTokens() {
    return tokenTracking.getOutputTokenCount(GenerationState.REASONING);
  }

  public int getToolsTokens() {
    return tokenTracking.getOutputTokenCount(GenerationState.TOOLS);
  }

  public int getTotalTokenCount() {
    return tokenTracking.getTotalTokenCount();
  }
}
