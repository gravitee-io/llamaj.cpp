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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_tokenize;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Low-level runner for native decision models: it evaluates already-rendered prompts and returns
 * one raw score per option. No token is generated.
 * <p>
 * One subclass per family, each with its own factory and prompt type. Code that knows the family
 * uses the subclass and its typed {@code scoreAll}; code that does not holds a plain
 * {@code LlamaDecider} from {@link #create} and calls {@link #score(List)}, which checks each
 * prompt's kind at run time:
 * <ul>
 *   <li>{@link LabelDecider} ({@link TextPrompt}) — openjev, lev, nimble, or any model with
 *       caller-chosen codes ({@link LabelDecider#custom}): logits of the label tokens at the last
 *       prompt token, with the common prefix of the prompts decoded once</li>
 *   <li>{@link KevDecider} ({@link TextPrompt}) — scaled dot product of the last token's hidden
 *       state with the hidden state at each option marker</li>
 *   <li>{@link LayaDecider} ({@link LayaPrompt}) — one column of the hidden state at each option
 *       marker</li>
 *   <li>{@link ClefDecider} ({@link ClefPrompt}) — a joint head over tokens tagged with a
 *       {@link DecisionOrder}</li>
 * </ul>
 * Only what needs the native model lives here — the family, its special tokens, the temperatures
 * and the {@code "systemone"} template read from the GGUF, and the forward pass. Rendering the
 * template and turning scores into answers are left to the caller.
 *
 * <pre>{@code
 * try (var decider = LlamaDecider.create(arena, model, LlamaDecider.Options.defaults())) {
 *   String tmpl = decider.template();               // render it with a Jinja engine
 *   List<float[]> scores = decider.score(List.of(
 *       TextPrompt.of(promptQ1, 3),
 *       TextPrompt.of(promptQ2, 2)));
 * }
 * }</pre>
 *
 * <h2>Resource management</h2>
 * The caller owns the {@link LlamaModel}; {@link #free()} releases only the internal context.
 *
 * <h2>Thread safety</h2>
 * Not thread-safe.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public abstract sealed class LlamaDecider
  implements Freeable, AutoCloseable
  permits LabelDecider, KevDecider, LayaDecider, ClefDecider {

  final LlamaModel model;
  final LlamaVocab vocab;
  final LlamaContext context;
  private final DecisionType type;
  private final Map<String, Float> temperatures;
  private final String template;

  private boolean freed = false;

  LlamaDecider(
    Arena arena,
    LlamaModel model,
    DecisionType type,
    LlamaContextParams params
  ) {
    this.model = model;
    this.type = type;
    this.vocab = new LlamaVocab(model);
    this.temperatures = Collections.unmodifiableMap(
      readTemperatures(arena, model)
    );
    this.template = new LlamaTemplate(
      arena,
      model,
      "systemone"
    ).templateString();
    this.context = new LlamaContext(arena, model, params);
  }

  /**
   * Creates the decider for the model's decision family, as the family's own factory would
   * ({@link LabelDecider#create}, {@link KevDecider#create}, {@link LayaDecider#create},
   * {@link ClefDecider#create}).
   *
   * @param arena   Arena used for the context params and metadata strings
   * @param model   Pre-loaded model (not freed on {@link #close()})
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @throws LlamaException if the model is not a decision model, or of an unknown family
   */
  public static LlamaDecider create(
    Arena arena,
    LlamaModel model,
    Options options
  ) {
    return switch (familyOf(arena, model)) {
      case OPENJEV, LEV, NIMBLE -> LabelDecider.create(arena, model, options);
      case KEV -> KevDecider.create(arena, model, options);
      case LAYA -> LayaDecider.create(arena, model, options);
      case CLEF -> ClefDecider.create(arena, model, options);
      case CUSTOM -> throw new IllegalStateException("CUSTOM is never read");
    };
  }

  /**
   * Scores each prompt. Each must be of this decider's prompt kind; the subclass's typed
   * {@code scoreAll} checks that at compile time instead.
   *
   * @param prompts The prompts
   * @return One {@code float[prompt.nOptions()]} of raw scores per prompt, in input order
   * @throws LlamaException if a prompt is of another kind
   */
  public abstract List<float[]> score(List<? extends DecisionPrompt> prompts);

  /** Scores one prompt; see {@link #score(List)}. */
  public float[] score(DecisionPrompt prompt) {
    return score(List.of(prompt)).getFirst();
  }

  /** The decision family declared by the model. */
  public DecisionType type() {
    checkNotFreed();
    return type;
  }

  /**
   * The calibration temperatures of the model ({@code <arch>.decision.temperature.<suffix>}), keyed
   * by suffix: a question type ({@code "choice"}) or a type and an option-count bucket
   * ({@code "choice.3_5"}). Empty if the model declares none.
   */
  public Map<String, Float> temperatures() {
    checkNotFreed();
    return temperatures;
  }

  /** The {@code "systemone"} Jinja prompt template of the model, or {@code null} if it has none. */
  public String template() {
    checkNotFreed();
    return template;
  }

  /**
   * Tokenizes text without adding BOS/EOS.
   *
   * @param text         The text
   * @param parseSpecial Whether special-token text (e.g. {@code <|im_start|>}) maps to its token
   */
  public int[] tokenize(String text, boolean parseSpecial) {
    checkNotFreed();
    try (Arena local = Arena.ofConfined()) {
      return tokenize(local, text, parseSpecial);
    }
  }

  /** Exposes the underlying context for advanced use cases. */
  public LlamaContext context() {
    checkNotFreed();
    return context;
  }

  /** Exposes the underlying model. The caller is responsible for freeing it. */
  public LlamaModel model() {
    checkNotFreed();
    return model;
  }

  @Override
  public void free() {
    if (freed) return;
    freed = true;
    context.free();
  }

  @Override
  public boolean isFree() {
    return freed;
  }

  @Override
  public void close() {
    free();
  }

  /* ------------------------------- for the subclasses ------------------------------- */

  /**
   * Context params for the families that read the embeddings output: one prompt per forward pass,
   * every token an output, the whole prompt in one ubatch.
   */
  static LlamaContextParams embeddingParams(Arena arena, Options options) {
    int nBatch = options.nBatch() != null ? options.nBatch() : 2048;
    return new LlamaContextParams(arena)
      .nCtx(options.nCtx() != null ? options.nCtx() : 8192)
      .embeddings(true)
      .poolingType(PoolingType.NONE)
      .nBatch(nBatch)
      .nUBatch(nBatch)
      .nOutputsMax(nBatch)
      .nSeqMax(1);
  }

  /** The model's decision family, read from its metadata. */
  static DecisionType familyOf(Arena arena, LlamaModel model) {
    return DecisionType.of(arena, model).orElseThrow(() ->
      new LlamaException(
        "not a decision model: no <arch>.decision.type metadata"
      )
    );
  }

  /** The model's decision family, which must be one of {@code allowed}. */
  static DecisionType requireFamily(
    Arena arena,
    LlamaModel model,
    Set<DecisionType> allowed
  ) {
    var type = familyOf(arena, model);
    if (!allowed.contains(type)) {
      throw new LlamaException(
        "the model is a %s decision model, not one of %s".formatted(
          type.metaName(),
          allowed
        )
      );
    }
    return type;
  }

  /** The prompts, each checked to be of {@code kind}. */
  static <P extends DecisionPrompt> List<P> requireKind(
    List<? extends DecisionPrompt> prompts,
    Class<P> kind
  ) {
    return prompts
      .stream()
      .map(prompt -> {
        if (!kind.isInstance(prompt)) {
          throw new LlamaException(
            "this decider takes %s, not %s".formatted(
              kind.getSimpleName(),
              prompt.getClass().getSimpleName()
            )
          );
        }
        return kind.cast(prompt);
      })
      .toList();
  }

  /** The tokens of a prompt, tokenizing its text with special-token parsing. */
  int[] tokensOf(TextPrompt prompt) {
    int[] tokens = prompt.tokens() != null
      ? prompt.tokens()
      : tokenize(prompt.text(), true);
    if (tokens.length == 0) {
      throw new LlamaException("a prompt must have at least one token");
    }
    return tokens;
  }

  /** Decodes one prompt in sequence 0 with every token an output; it must fit in one batch. */
  void decodeAllOutputs(int[] prompt) {
    checkFitsOneBatch(prompt.length);
    context.getMemory().clear();
    try (var local = Arena.ofConfined()) {
      var batch = new LlamaBatch(local, prompt.length, 0, 1);
      try {
        var seq = List.of(0);
        IntStream.range(0, prompt.length).forEach(k ->
          batch.add(prompt[k], k, seq, true)
        );
        int ret = context.decode(batch);
        if (ret != 0) {
          throw new LlamaException(
            "decode() returned non-zero status: %d".formatted(ret)
          );
        }
      } finally {
        batch.free();
      }
    }
  }

  void checkFitsOneBatch(int nTokens) {
    if (nTokens > context.nBatch()) {
      throw new LlamaException(
        "the prompt has %d tokens, it must fit in one batch (nBatch=%d)".formatted(
          nTokens,
          context.nBatch()
        )
      );
    }
  }

  /** Tokenizes without adding BOS/EOS, the way llama-server's decision code does. */
  int[] tokenize(Arena arena, String text, boolean parseSpecial) {
    var textSeg = arena.allocateFrom(text);
    int textLen = (int) textSeg.byteSize() - 1; // byte length without the NUL
    int n = -llama_tokenize(
      vocab.segment,
      textSeg,
      textLen,
      MemorySegment.NULL,
      0,
      false,
      parseSpecial
    );
    if (n <= 0) {
      return new int[0];
    }
    var buf = arena.allocate(JAVA_INT, n);
    if (
      llama_tokenize(
        vocab.segment,
        textSeg,
        textLen,
        buf,
        n,
        false,
        parseSpecial
      ) <
      0
    ) {
      throw new LlamaException("failed to tokenize");
    }
    return buf.toArray(JAVA_INT);
  }

  /** Frees the context and rethrows: for a subclass constructor that fails after super(). */
  RuntimeException failConstruction(RuntimeException e) {
    free();
    return e;
  }

  void checkNotFreed() {
    if (freed) {
      throw new LlamaException(
        "%s has been freed".formatted(getClass().getSimpleName())
      );
    }
  }

  private static Map<String, Float> readTemperatures(
    Arena arena,
    LlamaModel model
  ) {
    var prefix =
      model.metaVal(arena, "general.architecture") + ".decision.temperature.";
    return model
      .meta(arena)
      .entrySet()
      .stream()
      .filter(e -> e.getKey().startsWith(prefix))
      .collect(
        Collectors.toMap(
          e -> e.getKey().substring(prefix.length()),
          LlamaDecider::temperature,
          (a, b) -> a,
          LinkedHashMap::new
        )
      );
  }

  private static float temperature(Map.Entry<String, String> entry) {
    float temp;
    try {
      temp = Float.parseFloat(entry.getValue());
    } catch (NumberFormatException | NullPointerException _) {
      temp = 0;
    }
    if (!(temp > 0)) {
      throw new LlamaException(
        "invalid decision temperature: %s = %s".formatted(
          entry.getKey(),
          entry.getValue()
        )
      );
    }
    return temp;
  }

  /**
   * Configuration for {@link LlamaDecider}. Any {@code null} field takes a default.
   *
   * @param nCtx    Context size in tokens; {@code null} -> 8192
   * @param nBatch  Batch size in tokens; {@code null} -> llama.cpp's default for {@link LabelDecider},
   *                2048 for the other families, where it is also the ubatch size and the prompt
   *                limit
   * @param nSeqMax {@link LabelDecider} only: sequences per context; {@code null} -> 8. Sequence 0
   *                holds the shared prefix, so up to {@code nSeqMax - 1} prompts are decoded
   *                together. {@code 1} disables prefix sharing.
   */
  public record Options(Integer nCtx, Integer nBatch, Integer nSeqMax) {
    public static Options defaults() {
      return new Options(null, null, null);
    }

    public Options withNCtx(int nCtx) {
      return new Options(nCtx, nBatch, nSeqMax);
    }

    public Options withNBatch(int nBatch) {
      return new Options(nCtx, nBatch, nSeqMax);
    }

    public Options withNSeqMax(int nSeqMax) {
      return new Options(nCtx, nBatch, nSeqMax);
    }
  }
}
