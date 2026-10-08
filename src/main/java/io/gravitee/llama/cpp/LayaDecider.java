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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_vocab_mask;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_vocab_sep;

import java.lang.foreign.Arena;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Decider for {@link DecisionType#LAYA}: each option starts with a {@link #markerToken()} (the mask
 * token); its score is the {@link QuestionType#column()} of the hidden state at that marker.
 * Options are scored in prompt order. Prompts are {@link LayaPrompt}s, which carry their
 * {@link QuestionType}.
 * <p>
 * The model was trained on {@code [cls] question [sep] ([mask] option)* [sep] state [sep]}, with the
 * question and options cut to {@link #maxHeadTokens()}; the caller lays the tokens out that way
 * ({@link #sepToken()} is the separator). One prompt per forward pass; a prompt must fit in one
 * batch.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LayaDecider extends LlamaDecider {

  private final int markerToken;
  private final int sepToken;
  private final int maxHeadTokens;

  private LayaDecider(Arena arena, LlamaModel model, Options options) {
    super(arena, model, DecisionType.LAYA, embeddingParams(arena, options));
    try {
      this.markerToken = llama_vocab_mask(vocab.segment);
      this.sepToken = llama_vocab_sep(vocab.segment);
      if (markerToken < 0 || sepToken < 0) {
        throw new LlamaException("decision model has no mask or sep token");
      }
      var arch = model.metaVal(arena, "general.architecture");
      this.maxHeadTokens = Optional.ofNullable(
        model.metaVal(arena, arch + ".decision.max_head_tokens")
      )
        .flatMap(LayaDecider::parsePositive)
        .orElseThrow(() ->
          new LlamaException("decision model has no valid max_head_tokens")
        );
    } catch (RuntimeException e) {
      throw failConstruction(e);
    }
  }

  /**
   * Creates the decider of a laya model.
   *
   * @param arena   Arena used for the context params and metadata strings
   * @param model   Pre-loaded model (not freed on {@link #close()})
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @throws LlamaException if the model is not a laya model
   */
  public static LayaDecider create(
    Arena arena,
    LlamaModel model,
    Options options
  ) {
    requireFamily(arena, model, Set.of(DecisionType.LAYA));
    return new LayaDecider(arena, model, options);
  }

  /** The token that starts an option (the mask token). */
  public int markerToken() {
    checkNotFreed();
    return markerToken;
  }

  /** The separator token. */
  public int sepToken() {
    checkNotFreed();
    return sepToken;
  }

  /**
   * The token budget of the prompt head (question + options) the model was trained with
   * ({@code <arch>.decision.max_head_tokens}).
   */
  public int maxHeadTokens() {
    checkNotFreed();
    return maxHeadTokens;
  }

  @Override
  public List<float[]> score(List<? extends DecisionPrompt> prompts) {
    return scoreAll(requireKind(prompts, LayaPrompt.class));
  }

  /**
   * Scores each prompt.
   *
   * @param prompts The prompts
   * @return One {@code float[prompt.nOptions()]} of raw scores per prompt, in input order
   */
  public List<float[]> scoreAll(List<LayaPrompt> prompts) {
    checkNotFreed();
    return prompts.stream().map(this::scoreOne).toList();
  }

  private float[] scoreOne(LayaPrompt prompt) {
    int column = prompt.type().column();
    if (column >= model.nEmbdOut()) {
      throw new LlamaException(
        "column %d is out of the output".formatted(column)
      );
    }
    int[] tokens = prompt.tokens();
    if (tokens.length == 0) {
      throw new LlamaException("a prompt must have at least one token");
    }
    int[] markers = IntStream.range(0, tokens.length)
      .filter(i -> tokens[i] == markerToken)
      .toArray();
    if (markers.length != prompt.nOptions()) {
      throw new LlamaException(
        "the laya prompt has %d option markers for %d options".formatted(
          markers.length,
          prompt.nOptions()
        )
      );
    }
    decodeAllOutputs(tokens);

    float[] scores = new float[markers.length];
    IntStream.range(0, markers.length).forEach(o ->
      scores[o] = context.getEmbeddingsIth(markers[o])[column]
    );
    return scores;
  }

  private static Optional<Integer> parsePositive(String value) {
    try {
      int n = Integer.parseInt(value.trim());
      return n > 0 ? Optional.of(n) : Optional.empty();
    } catch (NumberFormatException _) {
      return Optional.empty();
    }
  }
}
