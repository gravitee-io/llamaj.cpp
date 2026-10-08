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

import java.lang.foreign.Arena;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Decider for {@link DecisionType#KEV}: each option ends with a {@link #markerToken()}
 * ({@code <|box_end|>}); its score is the dot product of the last token's hidden state (first half)
 * with the marker's hidden state (second half), divided by the square root of the half size.
 * Options are scored in prompt order. Prompts are {@link TextPrompt}s. One prompt per forward
 * pass; a prompt must fit in one batch.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class KevDecider extends LlamaDecider {

  private final int markerToken;

  private KevDecider(Arena arena, LlamaModel model, Options options) {
    super(arena, model, DecisionType.KEV, embeddingParams(arena, options));
    try {
      int[] box = tokenize(arena, "<|" + "box_end|>", true);
      if (box.length != 1) {
        throw new LlamaException("decision model has no <|box_end|> token");
      }
      this.markerToken = box[0];
    } catch (RuntimeException e) {
      throw failConstruction(e);
    }
  }

  /**
   * Creates the decider of a kev model.
   *
   * @param arena   Arena used for the context params and metadata strings
   * @param model   Pre-loaded model (not freed on {@link #close()})
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @throws LlamaException if the model is not a kev model
   */
  public static KevDecider create(
    Arena arena,
    LlamaModel model,
    Options options
  ) {
    requireFamily(arena, model, Set.of(DecisionType.KEV));
    return new KevDecider(arena, model, options);
  }

  /** The token that ends an option ({@code <|box_end|>}). */
  public int markerToken() {
    checkNotFreed();
    return markerToken;
  }

  @Override
  public List<float[]> score(List<? extends DecisionPrompt> prompts) {
    return scoreAll(requireKind(prompts, TextPrompt.class));
  }

  /**
   * Scores each prompt.
   *
   * @param prompts The prompts
   * @return One {@code float[prompt.nOptions()]} of raw scores per prompt, in input order
   */
  public List<float[]> scoreAll(List<TextPrompt> prompts) {
    checkNotFreed();
    return prompts.stream().map(this::scoreOne).toList();
  }

  private float[] scoreOne(TextPrompt prompt) {
    int[] tokens = tokensOf(prompt);
    int[] markers = IntStream.range(0, tokens.length)
      .filter(i -> tokens[i] == markerToken)
      .toArray();
    if (markers.length != prompt.nOptions()) {
      throw new LlamaException(
        "the kev prompt has %d option markers for %d options".formatted(
          markers.length,
          prompt.nOptions()
        )
      );
    }
    decodeAllOutputs(tokens);

    int half = model.nEmbdOut() / 2;
    float scale = (float) Math.sqrt(half);
    float[] query = context.getEmbeddingsIth(tokens.length - 1);
    float[] scores = new float[markers.length];
    IntStream.range(0, markers.length).forEach(m -> {
      float[] option = context.getEmbeddingsIth(markers[m]);
      float dot = 0f;
      for (int k = 0; k < half; k++) {
        dot += query[k] * option[half + k];
      }
      scores[m] = dot / scale;
    });
    return scores;
  }
}
