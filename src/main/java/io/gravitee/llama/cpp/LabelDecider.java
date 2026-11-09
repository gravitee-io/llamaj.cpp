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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_get_logits_ith;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Decider for the label-token families ({@link DecisionType#OPENJEV}, {@link DecisionType#LEV},
 * {@link DecisionType#NIMBLE}, {@link DecisionType#PPLX_DECIDER}, {@link DecisionType#VLLM_D3},
 * {@link DecisionType#GRANITE_SWITCH}, {@link DecisionType#H2O_LIGHTNING}, see {@link #create})
 * and for any causal model with caller-chosen codes ({@link DecisionType#CUSTOM}, see
 * {@link #custom}): the score of option {@code i} is the logit of {@code labels().get(i)} at the
 * last prompt token. Prompts are {@link TextPrompt}s, or {@link MediaPrompt}s for a decider created
 * with an {@link MtmdContext}.
 *
 * <h2>Shared prefix</h2>
 * The prompts of one {@link #score} call usually differ only near their end (a nimble prompt lists
 * every question and names the asked one last). Their common prefix is decoded once in sequence 0,
 * copied to sequences {@code 1..nSeqMax-1} with {@link LlamaMemory#seqCp}, and only the tails are
 * decoded, packed into shared batches.
 *
 * <h2>Non-causal models</h2>
 * A model whose attention reads the prompt both ways ({@link #causal()} is false) has no prefix to
 * share: each prompt is decoded on its own, in one ubatch, so it must fit in one batch. Such a model
 * takes text prompts only.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LabelDecider extends LlamaDecider {

  /** Most label codes a lev/nimble model uses (llama-server's limit). */
  private static final int MAX_LABELS = 255;

  private static final String LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  private static final Set<DecisionType> FAMILIES = EnumSet.of(
    DecisionType.OPENJEV,
    DecisionType.LEV,
    DecisionType.NIMBLE,
    DecisionType.PPLX_DECIDER,
    DecisionType.VLLM_D3,
    DecisionType.GRANITE_SWITCH,
    DecisionType.H2O_LIGHTNING
  );

  /**
   * A label: the code written in the prompt for an option, and the single token whose logit scores
   * it.
   */
  public record Label(String code, int token) {}

  /** Where to read the logits of one prompt: its index in the request and its row in the batch. */
  private record Output(int index, int row) {}

  /**
   * Tokens {@code [from, to)} of a prompt to decode in sequence {@code seqId} at their own
   * positions; {@code index < 0} decodes without reading logits.
   */
  private record Job(int index, int seqId, int[] prompt, int from, int to) {
    Job(int index, int seqId, int[] prompt, int from) {
      this(index, seqId, prompt, from, prompt.length);
    }
  }

  private final List<Label> labels;

  private LabelDecider(
    Arena arena,
    LlamaModel model,
    DecisionType type,
    MtmdContext mtmd,
    Options options
  ) {
    super(arena, model, type, params(arena, options, isCausal(model)), mtmd);
    try {
      this.labels = withScratch(scratch -> familyLabels(scratch, type));
    } catch (RuntimeException e) {
      throw failConstruction(e);
    }
  }

  private LabelDecider(
    Arena arena,
    LlamaModel model,
    MtmdContext mtmd,
    Options options,
    String context,
    List<String> codes
  ) {
    super(
      arena,
      model,
      DecisionType.CUSTOM,
      params(arena, options, isCausal(model)),
      mtmd
    );
    try {
      this.labels = withScratch(scratch ->
        customLabels(scratch, context, codes)
      );
    } catch (RuntimeException e) {
      throw failConstruction(e);
    }
  }

  /**
   * Creates the decider of an openjev, lev, nimble, pplx-decider, vllm-d3, granite-switch or
   * h2o-lightning model, for text prompts.
   *
   * @param arena   Arena that owns the decider's context
   * @param model   Pre-loaded model (not freed on {@link #close()})
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @throws LlamaException if the model is not of one of these families
   */
  public static LabelDecider create(
    Arena arena,
    LlamaModel model,
    Options options
  ) {
    return create(arena, model, null, options);
  }

  /**
   * Same as {@link #create(Arena, LlamaModel, Options)}, with the multimodal context that
   * {@link MediaPrompt}s are tokenized and evaluated with.
   *
   * @param arena   Arena that owns the decider's context
   * @param model   Pre-loaded model (not freed on {@link #close()})
   * @param mtmd    The model's multimodal context (caller-owned), or {@code null} for text only
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @throws LlamaException if the model is not of one of these families
   */
  public static LabelDecider create(
    Arena arena,
    LlamaModel model,
    MtmdContext mtmd,
    Options options
  ) {
    var type = requireFamily(model, FAMILIES);
    return new LabelDecider(arena, model, type, mtmd, options);
  }

  /**
   * Turns any causal model into a decision model scored on caller-chosen label tokens: option
   * {@code i} scores the logit of {@code codes.get(i)} at the last prompt token. The prompt must end
   * where the code is the next token — e.g. {@code Answer: (} for {@code "A"}, or a
   * <code>&#123;"sentiment": "</code> prefill for {@code "positive"}.
   * <p>
   * Each code is tokenized on its own. That matches the token the model predicts on BPE vocabs
   * (Qwen, Llama 3, Gemma); on SentencePiece vocabs with {@code add_space_prefix} (Llama 2,
   * Mistral) a standalone {@code "A"} becomes {@code "▁A"}, a different token. Use
   * {@link #custom(Arena, LlamaModel, Options, String, List)} there, which resolves each code where
   * the prompt ends.
   *
   * @param arena   Arena that owns the decider's context
   * @param model   Pre-loaded model (not freed on {@link #close()}); it needs no decision metadata
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @param codes   The label text of each option, in option order; each must be one distinct token
   * @throws LlamaException if a code is not a single token, or two codes share a token
   */
  public static LabelDecider custom(
    Arena arena,
    LlamaModel model,
    Options options,
    List<String> codes
  ) {
    return new LabelDecider(arena, model, null, options, null, codes);
  }

  /**
   * Same as {@link #custom(Arena, LlamaModel, Options, List)}, with each code resolved where the
   * prompt ends: the code's token is the one {@code context + code} adds after the tokens of
   * {@code context}. Pass the tail of the rendered prompt (e.g. {@code "Answer: ("} or
   * <code>&#123;"sentiment": "</code>); this is the reliable form on every vocab type.
   *
   * @param arena   Arena that owns the decider's context
   * @param model   Pre-loaded model (not freed on {@link #close()}); it needs no decision metadata
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @param context The text the prompts end with, tokenized with special-token parsing
   * @param codes   The label text of each option, in option order
   * @throws LlamaException if a code does not add exactly one token after {@code context}, or two
   *                        codes share a token
   */
  public static LabelDecider custom(
    Arena arena,
    LlamaModel model,
    Options options,
    String context,
    List<String> codes
  ) {
    return custom(arena, model, null, options, context, codes);
  }

  /**
   * Same as {@link #custom(Arena, LlamaModel, Options, String, List)}, with the multimodal context
   * that {@link MediaPrompt}s are tokenized and evaluated with.
   *
   * @param arena   Arena that owns the decider's context
   * @param model   Pre-loaded model (not freed on {@link #close()}); it needs no decision metadata
   * @param mtmd    The model's multimodal context (caller-owned), or {@code null} for text only
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @param context The text the prompts end with, tokenized with special-token parsing
   * @param codes   The label text of each option, in option order
   * @throws LlamaException if a code does not add exactly one token after {@code context}, or two
   *                        codes share a token
   */
  public static LabelDecider custom(
    Arena arena,
    LlamaModel model,
    MtmdContext mtmd,
    Options options,
    String context,
    List<String> codes
  ) {
    Objects.requireNonNull(context, "context");
    return new LabelDecider(arena, model, mtmd, options, context, codes);
  }

  private static LlamaContextParams params(
    Arena arena,
    Options options,
    boolean causal
  ) {
    if (!causal) {
      int nBatch = options.nBatch() != null ? options.nBatch() : 2048;
      return new LlamaContextParams(arena)
        .nCtx(options.nCtx() != null ? options.nCtx() : 8192)
        .nBatch(nBatch)
        .nUBatch(nBatch)
        .nSeqMax(1);
    }
    // A unified KV cache lets every sequence see the shared prefix cells without copying data,
    // and gives each sequence the whole nCtx instead of nCtx / nSeqMax.
    var params = new LlamaContextParams(arena)
      .nCtx(options.nCtx() != null ? options.nCtx() : 8192)
      .kvUnified(true)
      .nSeqMax(options.nSeqMax() != null ? options.nSeqMax() : 8);
    if (options.nBatch() != null) params.nBatch(options.nBatch());
    return params;
  }

  /**
   * The label codes of the model and their tokens, in option order (option {@code i} is
   * {@code labels().get(i)}):
   * <ul>
   *   <li>openjev: {@code A..Z a..z}, all single tokens</li>
   *   <li>lev, nimble, pplx-decider, vllm-d3, granite-switch: {@code A..Z} then {@code AA..ZZ},
   *       only the codes that are a single token, at most 255</li>
   *   <li>h2o-lightning: the label texts of the GGUF ({@code <arch>.decision.labels}), each a
   *       single distinct token</li>
   *   <li>custom: the codes given to {@link #custom}</li>
   * </ul>
   */
  public List<Label> labels() {
    checkNotFreed();
    return labels;
  }

  @Override
  public List<float[]> score(List<? extends DecisionPrompt> prompts) {
    if (prompts.stream().allMatch(TextPrompt.class::isInstance)) {
      return scoreAll(requireKind(prompts, TextPrompt.class));
    }
    return prompts
      .stream()
      .map(prompt ->
        switch (prompt) {
          case TextPrompt text -> scoreAll(List.of(text)).getFirst();
          case MediaPrompt media -> scoreMedia(List.of(media)).getFirst();
          default -> throw new LlamaException(
            "this decider takes TextPrompt or MediaPrompt, not %s".formatted(
              prompt.getClass().getSimpleName()
            )
          );
        }
      )
      .toList();
  }

  /**
   * Scores each prompt.
   *
   * @param prompts The prompts
   * @return One {@code float[prompt.nOptions()]} of raw scores per prompt, in input order
   */
  public List<float[]> scoreAll(List<TextPrompt> prompts) {
    checkNotFreed();
    if (prompts.isEmpty()) {
      return List.of();
    }
    checkOptions(
      prompts.stream().mapToInt(TextPrompt::nOptions).max().orElseThrow()
    );

    int maxOptions = prompts
      .stream()
      .mapToInt(TextPrompt::nOptions)
      .max()
      .orElseThrow();
    int[] labelTokens = labels
      .stream()
      .limit(maxOptions)
      .mapToInt(Label::token)
      .toArray();
    float[][] logits = scoreTokens(
      prompts.stream().map(this::tokensOf).toList(),
      labelTokens
    );
    return IntStream.range(0, prompts.size())
      .mapToObj(i -> Arrays.copyOf(logits[i], prompts.get(i).nOptions()))
      .toList();
  }

  /**
   * Scores each media prompt, one forward pass each (no prefix sharing). The decider must have been
   * created with an {@link MtmdContext}, and the model must be causal.
   *
   * @param prompts The prompts
   * @return One {@code float[prompt.nOptions()]} of raw scores per prompt, in input order
   */
  public List<float[]> scoreMedia(List<MediaPrompt> prompts) {
    checkNotFreed();
    return prompts
      .stream()
      .map(prompt -> {
        checkOptions(prompt.nOptions());
        decodeMedia(prompt);
        float[] row = logitsRow(-1);
        float[] scores = new float[prompt.nOptions()];
        IntStream.range(0, scores.length).forEach(i ->
          scores[i] = row[labels.get(i).token()]
        );
        return scores;
      })
      .toList();
  }

  private void checkOptions(int nOptions) {
    if (nOptions > labels.size()) {
      throw new LlamaException(
        "too many options (%d), this model supports at most %d".formatted(
          nOptions,
          labels.size()
        )
      );
    }
  }

  private float[] logitsRow(int row) {
    var logits = llama_get_logits_ith(context.segment, row);
    if (logits == null || logits.address() == 0) {
      throw new LlamaException("failed to get logits");
    }
    return logits
      .reinterpret(vocab.nVocab() * JAVA_FLOAT.byteSize())
      .toArray(JAVA_FLOAT);
  }

  /** The logits of {@code tokens} at the last token of each prompt. */
  private float[][] scoreTokens(List<int[]> prompts, int[] tokens) {
    int n = prompts.size();
    float[][] results = new float[n][];
    if (!causal) {
      prompts.forEach(prompt -> checkFitsOneBatch(prompt.length));
    }
    int[] first = prompts.getFirst();
    // Shared prefix, leaving each prompt at least one token of its own to read logits from.
    int shared = prompts
      .stream()
      .mapToInt(prompt ->
        Math.min(commonPrefix(first, prompt), prompt.length - 1)
      )
      .min()
      .orElseThrow();
    int nGroup = context.nSeqMax() - 1;
    var memory = context.getMemory();

    try (var local = Arena.ofConfined()) {
      var batch = new LlamaBatch(local, context.nBatch(), 0, 1);
      try {
        if (n == 1 || shared == 0 || nGroup < 1 || !causal) {
          // No sharing: one prompt at a time in sequence 0.
          IntStream.range(0, n).forEach(i -> {
            memory.clear();
            decode(
              batch,
              List.of(new Job(i, 0, prompts.get(i), 0)),
              tokens,
              results
            );
          });
          return results;
        }
        memory.clear();
        decode(
          batch,
          List.of(new Job(-1, 0, first, 0, shared)),
          tokens,
          results
        );
        for (int g = 0; g < n; g += nGroup) {
          int from = g;
          var jobs = IntStream.range(from, Math.min(n, from + nGroup))
            .mapToObj(i -> new Job(i, i - from + 1, prompts.get(i), shared))
            .toList();
          jobs.forEach(job -> {
            memory.seqRm(job.seqId(), -1, -1);
            memory.seqCp(0, job.seqId(), -1, -1);
          });
          decode(batch, jobs, tokens, results);
        }
        return results;
      } finally {
        batch.free();
      }
    }
  }

  /** Packs the jobs into batches of at most nBatch tokens, and reads the logits of each last token. */
  private void decode(
    LlamaBatch batch,
    List<Job> jobs,
    int[] tokens,
    float[][] results
  ) {
    int nBatch = context.nBatch();
    var outputs = new ArrayList<Output>();
    int nTokens = 0;
    batch.clear();
    for (var job : jobs) {
      var seq = List.of(job.seqId());
      for (int k = job.from(); k < job.to(); k++) {
        boolean last = job.index() >= 0 && k == job.to() - 1;
        batch.add(job.prompt()[k], k, seq, last);
        if (last) {
          outputs.add(new Output(job.index(), nTokens));
        }
        if (++nTokens == nBatch) {
          flush(batch, outputs, tokens, results);
          outputs.clear();
          nTokens = 0;
        }
      }
    }
    if (nTokens > 0) {
      flush(batch, outputs, tokens, results);
    }
  }

  private void flush(
    LlamaBatch batch,
    List<Output> outputs,
    int[] tokens,
    float[][] results
  ) {
    int ret = context.decode(batch);
    if (ret != 0) {
      throw new LlamaException(
        "decode() returned non-zero status: %d".formatted(ret)
      );
    }
    long logitsBytes = vocab.nVocab() * JAVA_FLOAT.byteSize();
    for (var output : outputs) {
      var logits = llama_get_logits_ith(context.segment, output.row());
      if (logits == null || logits.address() == 0) {
        throw new LlamaException("failed to get logits");
      }
      var row = logits.reinterpret(logitsBytes);
      float[] scores = new float[tokens.length];
      IntStream.range(0, tokens.length).forEach(t ->
        scores[t] = row.getAtIndex(JAVA_FLOAT, tokens[t])
      );
      results[output.index()] = scores;
    }
    batch.clear();
  }

  private List<Label> customLabels(
    Arena arena,
    String context,
    List<String> codes
  ) {
    if (codes.isEmpty()) {
      throw new LlamaException("a custom decider needs at least one code");
    }
    var labels = codes
      .stream()
      .map(code ->
        new Label(
          code,
          context == null
            ? singleToken(arena, code)
            : tokenAfter(arena, context, code)
        )
      )
      .toList();
    var seen = new HashSet<Integer>();
    labels
      .stream()
      .filter(label -> !seen.add(label.token()))
      .findFirst()
      .ifPresent(label -> {
        throw new LlamaException(
          "code '%s' repeats another code's token".formatted(label.code())
        );
      });
    return labels;
  }

  private List<Label> familyLabels(Arena arena, DecisionType type) {
    return switch (type) {
      // one letter per option, each must be a single token
      case OPENJEV -> (LETTERS + LETTERS.toLowerCase()).chars()
        .mapToObj(Character::toString)
        .map(code -> new Label(code, singleToken(arena, code)))
        .toList();
      case H2O_LIGHTNING -> modelLabels(arena);
      // A..Z then AA..ZZ, only the codes that are a single token
      case LEV, NIMBLE, PPLX_DECIDER, VLLM_D3, GRANITE_SWITCH -> Stream.concat(
        LETTERS.chars().mapToObj(Character::toString),
        LETTERS.chars()
          .mapToObj(Character::toString)
          .flatMap(a ->
            LETTERS.chars().mapToObj(b -> a + Character.toString(b))
          )
      )
        .<Label>mapMulti((code, sink) -> {
          int[] tokens = tokenize(arena, code, false);
          if (tokens.length == 1) sink.accept(new Label(code, tokens[0]));
        })
        .limit(MAX_LABELS)
        .toList();
      default -> throw new IllegalArgumentException(
        "not a label family: %s".formatted(type)
      );
    };
  }

  private List<Label> modelLabels(Arena arena) {
    if (!LlamaExt.decisionLabelsAvailable()) {
      throw new LlamaException(
        "reading the decision labels needs llama_model_decision_label:%n%s".formatted(
          LlamaExt.decisionLabelsResolutionReport()
        )
      );
    }
    var codes = LlamaExt.decisionLabels(model);
    if (codes.isEmpty()) {
      throw new LlamaException("decision model has no labels");
    }
    var seen = new HashSet<Integer>();
    return codes
      .stream()
      .map(code -> {
        int[] tokens = tokenize(arena, code, false);
        if (tokens.length != 1 || !seen.add(tokens[0])) {
          throw new LlamaException(
            "decision label \"%s\" is not a single distinct token".formatted(
              code
            )
          );
        }
        return new Label(code, tokens[0]);
      })
      .toList();
  }

  /** The one token {@code context + code} adds after the tokens of {@code context}. */
  private int tokenAfter(Arena arena, String context, String code) {
    int[] before = tokenize(arena, context, true);
    int[] after = tokenize(arena, context + code, true);
    if (
      after.length != before.length + 1 ||
      Arrays.mismatch(before, after) != before.length
    ) {
      throw new LlamaException(
        "code '%s' does not add exactly one token after the context (%d -> %d tokens)".formatted(
          code,
          before.length,
          after.length
        )
      );
    }
    return after[before.length];
  }

  private int singleToken(Arena arena, String code) {
    int[] tokens = tokenize(arena, code, false);
    if (tokens.length != 1) {
      throw new LlamaException(
        "code '%s' is %d tokens, it must be one".formatted(code, tokens.length)
      );
    }
    return tokens[0];
  }

  private static int commonPrefix(int[] a, int[] b) {
    int mismatch = Arrays.mismatch(a, b);
    return mismatch < 0 ? a.length : mismatch;
  }
}
