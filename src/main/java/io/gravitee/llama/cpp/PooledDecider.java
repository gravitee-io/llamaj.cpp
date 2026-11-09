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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_pooling_type;

import java.lang.foreign.Arena;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Decider for {@link DecisionType#MINICPM_OCR_JEV}: a classifier whose scores are the pooled output
 * of the sequence, one value per class ({@link #classes()}), read with the pooling type of the
 * GGUF. Every prompt is scored into all the classes, so its {@code nOptions} must be
 * {@code classes().size()}.
 * <p>
 * Prompts are {@link TextPrompt}s, or {@link MediaPrompt}s for a decider created with an
 * {@link MtmdContext}. One prompt per forward pass; a text prompt must fit in one batch.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class PooledDecider extends LlamaDecider {

  private final List<String> classes;

  private PooledDecider(
    Arena arena,
    LlamaModel model,
    MtmdContext mtmd,
    Options options
  ) {
    super(
      arena,
      model,
      DecisionType.MINICPM_OCR_JEV,
      params(arena, options),
      mtmd
    );
    try {
      if (llama_pooling_type(context.segment) == PoolingType.NONE) {
        throw new LlamaException("decision model has no pooled output");
      }
      int nClasses = model.nClsOut();
      if (nClasses < 1 || nClasses != model.nEmbdOut()) {
        throw new LlamaException(
          "decision model must output one score per class, got %d classes and %d scores".formatted(
            nClasses,
            model.nEmbdOut()
          )
        );
      }
      this.classes = withScratch(scratch ->
        IntStream.range(0, nClasses)
          .mapToObj(i ->
            Objects.requireNonNullElse(model.clsLabel(scratch, i), "")
          )
          .toList()
      );
    } catch (RuntimeException e) {
      throw failConstruction(e);
    }
  }

  /**
   * Creates the decider of an openbmb-minicpm-ocr-jev model, for text prompts.
   *
   * @param arena   Arena that owns the decider's context
   * @param model   Pre-loaded model (not freed on {@link #close()})
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @throws LlamaException if the model is not of this family
   */
  public static PooledDecider create(
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
   * @throws LlamaException if the model is not of this family
   */
  public static PooledDecider create(
    Arena arena,
    LlamaModel model,
    MtmdContext mtmd,
    Options options
  ) {
    requireFamily(model, Set.of(DecisionType.MINICPM_OCR_JEV));
    return new PooledDecider(arena, model, mtmd, options);
  }

  private static LlamaContextParams params(Arena arena, Options options) {
    int nBatch = options.nBatch() != null ? options.nBatch() : 2048;
    return new LlamaContextParams(arena)
      .nCtx(options.nCtx() != null ? options.nCtx() : 8192)
      .embeddings(true)
      .poolingType(PoolingType.UNSPECIFIED)
      .nBatch(nBatch)
      .nUBatch(nBatch)
      .nOutputsMax(nBatch)
      .nSeqMax(1);
  }

  /** The classifier label of each score, in score order ({@code ""} for an unnamed class). */
  public List<String> classes() {
    checkNotFreed();
    return classes;
  }

  @Override
  public List<float[]> score(List<? extends DecisionPrompt> prompts) {
    checkNotFreed();
    return prompts
      .stream()
      .map(prompt ->
        switch (prompt) {
          case TextPrompt text -> scoreText(text);
          case MediaPrompt media -> scoreMedia(media);
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
   * Scores each text prompt.
   *
   * @param prompts The prompts
   * @return One {@code float[classes().size()]} of raw scores per prompt, in input order
   */
  public List<float[]> scoreAll(List<TextPrompt> prompts) {
    checkNotFreed();
    return prompts.stream().map(this::scoreText).toList();
  }

  /**
   * Scores each media prompt.
   *
   * @param prompts The prompts
   * @return One {@code float[classes().size()]} of raw scores per prompt, in input order
   */
  public List<float[]> scoreMedia(List<MediaPrompt> prompts) {
    checkNotFreed();
    return prompts.stream().map(this::scoreMedia).toList();
  }

  private float[] scoreText(TextPrompt prompt) {
    checkClasses(prompt.nOptions());
    decodeAllOutputs(tokensOf(prompt));
    return pooled();
  }

  private float[] scoreMedia(MediaPrompt prompt) {
    checkClasses(prompt.nOptions());
    decodeMedia(prompt);
    return pooled();
  }

  private void checkClasses(int nOptions) {
    if (nOptions != classes.size()) {
      throw new LlamaException(
        "the prompt asks for %d scores, this model outputs %d classes".formatted(
          nOptions,
          classes.size()
        )
      );
    }
  }

  private float[] pooled() {
    float[] scores = context.getEmbeddingsSeq(0);
    if (scores == null) {
      throw new LlamaException("failed to get the class scores");
    }
    return scores;
  }
}
