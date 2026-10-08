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

import static io.gravitee.llama.cpp.LlamaRuntime.LLAMA_PROCESS_TYPE_DECODE;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_batch_ext_add_token;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_batch_ext_free;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_batch_ext_init;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_batch_ext_set_output_embd;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_batch_ext_set_pos;
import static io.gravitee.llama.cpp.LlamaRuntime.llama_process;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.Arena;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Decider for {@link DecisionType#CLEF}: a joint head that reads every question and option span of
 * one prompt, tagged with a {@link DecisionOrder} per token ({@link ClefPrompt}), and
 * returns one score per {@link DecisionOrder#OPTION} span in prompt order. A NaN score means the
 * head could not use the order.
 * <p>
 * Spans of the same role must be separated by {@link DecisionOrder#NONE} tokens; an option span
 * belongs to the last question span before it. One prompt per forward pass; a prompt must fit in
 * one batch. Needs the staging symbol {@code llama_batch_ext_set_decision_order}
 * ({@link LlamaExt#decisionAvailable()}).
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class ClefDecider extends LlamaDecider {

  /**
   * Creates the decider of a clef model.
   *
   * @param arena   Arena used for the context params and metadata strings
   * @param model   Pre-loaded model (not freed on {@link #close()})
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   * @throws LlamaException if the model is not a clef model
   */
  public static ClefDecider create(
    Arena arena,
    LlamaModel model,
    Options options
  ) {
    requireFamily(arena, model, Set.of(DecisionType.CLEF));
    return new ClefDecider(arena, model, options);
  }

  private ClefDecider(Arena arena, LlamaModel model, Options options) {
    super(arena, model, DecisionType.CLEF, embeddingParams(arena, options));
  }

  @Override
  public List<float[]> score(List<? extends DecisionPrompt> prompts) {
    return scoreAll(requireKind(prompts, ClefPrompt.class));
  }

  /**
   * Scores each prompt.
   *
   * @param prompts The prompts
   * @return One {@code float[prompt.nOptions()]} of raw scores per prompt, in input order
   */
  public List<float[]> scoreAll(List<ClefPrompt> prompts) {
    checkNotFreed();
    if (!LlamaExt.decisionAvailable()) {
      throw new LlamaException(
        "clef needs llama_batch_ext_set_decision_order:%n%s".formatted(
          LlamaExt.decisionResolutionReport()
        )
      );
    }
    return prompts.stream().map(this::scoreOne).toList();
  }

  private float[] scoreOne(ClefPrompt prompt) {
    var order = prompt.order();
    int[] tokens = prompt.tokens();
    checkFitsOneBatch(tokens.length);
    context.getMemory().clear();

    var batch = llama_batch_ext_init(context.segment);
    if (batch == null || batch.address() == 0) {
      throw new LlamaException("llama_batch_ext_init failed");
    }
    try (var local = Arena.ofConfined()) {
      var pos = local.allocate(JAVA_INT);
      for (int k = 0; k < tokens.length; k++) {
        int idx = llama_batch_ext_add_token(batch, 0, tokens[k]);
        if (idx < 0) {
          throw new LlamaException(
            "llama_batch_ext_add_token failed: %d".formatted(idx)
          );
        }
        pos.set(JAVA_INT, 0, k);
        if (
          !llama_batch_ext_set_pos(batch, idx, pos) ||
          !llama_batch_ext_set_output_embd(batch, idx, true) ||
          !LlamaExt.setDecisionOrder(batch, idx, order[k])
        ) {
          throw new LlamaException(
            "failed to fill the clef batch at %d".formatted(k)
          );
        }
      }
      int ret = llama_process(
        context.segment,
        LLAMA_PROCESS_TYPE_DECODE,
        batch
      );
      if (ret != 0) {
        throw new LlamaException(
          "llama_process returned non-zero status: %d".formatted(ret)
        );
      }
    } finally {
      llama_batch_ext_free(batch);
    }

    // the scores are the first output rows, one value each
    float[] scores = new float[prompt.nOptions()];
    IntStream.range(0, scores.length).forEach(i ->
      scores[i] = context.getEmbeddingsIth(i)[0]
    );
    return scores;
  }
}
