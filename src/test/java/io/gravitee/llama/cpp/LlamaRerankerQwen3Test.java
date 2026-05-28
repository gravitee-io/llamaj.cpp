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

import static io.gravitee.llama.cpp.LlamaCppTest.RERANKER_MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.RERANKER_MODEL_TO_DOWNLOAD;
import static io.gravitee.llama.cpp.LlamaCppTest.getModelPath;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

/**
 * Integration tests for {@link LlamaReranker} using the Qwen3-Reranker-0.6B model
 * (qwen3 decoder arch, auto-detected format=QWEN3 since nClsOut=2).
 *
 * <p>Qwen3-Reranker returns {@code float[2]} with P(yes) at index 0 and P(no) at
 * index 1 (softmax applied inside the model).
 *
 * @author Remi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class LlamaRerankerQwen3Test extends LlamaCppTest {

  private static Arena arena;
  private static LlamaModel model;
  private static LlamaReranker reranker;

  // Broken across concatenations to avoid prettier-java lexing "|>" as a generic.
  private static final String IM_START = "<|" + "im_start|>";
  private static final String IM_END = "<|" + "im_end|>";

  /**
   * Qwen3-Reranker chat template: wraps query + document in system/user messages
   * instructing the model to answer yes/no. Provided as a test-local template to
   * demonstrate the {@link RerankTemplate} extension point - production callers
   * would typically pass a template from gravitee-inference.
   */
  private static final RerankTemplate QWEN3_TEMPLATE = (query, document) ->
    IM_START +
    "system\nJudge whether the Document is relevant to the Query. " +
    "Answer 'yes' or 'no'." +
    IM_END +
    "\n" +
    IM_START +
    "user\nQuery: " +
    query +
    "\nDocument: " +
    document +
    "\nRelevant:" +
    IM_END +
    "\n";

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

    var modelPath = getModelPath(
      RERANKER_MODEL_PATH,
      RERANKER_MODEL_TO_DOWNLOAD
    );
    model = new LlamaModel(arena, modelPath, new LlamaModelParams(arena));
    reranker = new LlamaReranker(
      arena,
      model,
      LlamaReranker.Options.defaults().withTemplate(QWEN3_TEMPLATE)
    );
  }

  @AfterAll
  static void afterAll() {
    if (reranker != null) reranker.close();
    if (model != null) model.free();
    LlamaRuntime.llama_backend_free();
    arena.close();
  }

  /** Extract the relevance probability: P(yes) is at index 0 for Qwen3. */
  private float relevance(String query, String doc) {
    return reranker.score(query, doc)[0];
  }

  @Test
  @Order(1)
  void template_should_be_the_configured_qwen3_template() {
    assertThat(reranker.template()).isSameAs(QWEN3_TEMPLATE);
  }

  @Test
  @Order(1)
  void defaults_should_resolve_nCtx_to_model_trained_context() {
    assertThat(reranker.context().nCtx()).isEqualTo(model.nCtxTrain());
  }

  @Test
  @Order(1)
  void defaults_should_keep_llama_cpp_defaults_for_nBatch_and_nSeqMax() {
    assertThat(reranker.context().nBatch()).isEqualTo(2048);
    assertThat(reranker.context().nSeqMax()).isEqualTo(1);
  }

  @Test
  @Order(2)
  void nClsOut_should_be_two_for_qwen3_reranker() {
    assertThat(reranker.nClsOut()).isEqualTo(2);
  }

  @Test
  @Order(3)
  void score_should_return_two_probabilities() {
    float[] scores = reranker.score(
      "capital of France",
      "Paris is the capital."
    );
    assertThat(scores).hasSize(2);
    // Softmax output: values are in [0,1] and approximately sum to 1
    double sum = scores[0] + scores[1];
    assertThat(sum).isCloseTo(1.0, org.assertj.core.data.Offset.offset(0.05));
  }

  @Test
  @Order(4)
  void score_should_be_deterministic() {
    String query = "capital of France";
    String doc = "Paris is the capital and largest city of France.";
    float[] first = reranker.score(query, doc);
    float[] second = reranker.score(query, doc);
    assertThat(first).containsExactly(second);
  }

  @Test
  @Order(5)
  void relevant_pair_should_score_higher_than_irrelevant_pair() {
    String query = "What is a panda?";
    String relevantDoc =
      "The giant panda is a bear species endemic to China, known for its distinctive black and white markings.";
    String irrelevantDoc =
      "The boiling point of water is 100 degrees Celsius at standard atmospheric pressure.";

    float relevantScore = relevance(query, relevantDoc);
    float irrelevantScore = relevance(query, irrelevantDoc);

    assertThat((double) relevantScore)
      .as(
        "relevant P(yes) (%.4f) should be higher than irrelevant P(yes) (%.4f)",
        relevantScore,
        irrelevantScore
      )
      .isGreaterThan(irrelevantScore);
  }

  @Test
  @Order(6)
  void exact_match_should_score_higher_than_unrelated_document() {
    String query = "speed of light";
    String exactDoc =
      "The speed of light in a vacuum is approximately 299,792,458 metres per second.";
    String unrelatedDoc =
      "Photosynthesis is the process by which plants convert sunlight into glucose using chlorophyll.";

    float exactScore = relevance(query, exactDoc);
    float unrelatedScore = relevance(query, unrelatedDoc);

    assertThat((double) exactScore)
      .as(
        "exact-match P(yes) (%.4f) should exceed unrelated P(yes) (%.4f)",
        exactScore,
        unrelatedScore
      )
      .isGreaterThan(unrelatedScore);
  }

  @Test
  @Order(7)
  void scoreAll_should_return_one_score_array_per_document() {
    var docs = java.util.List.of(
      "Paris is the capital of France.",
      "The Eiffel Tower is in Paris.",
      "Mitochondria generate ATP."
    );
    var results = reranker.scoreAll("What is the capital of France?", docs);
    assertThat(results).hasSize(3);
    for (float[] s : results) {
      assertThat(s).hasSize(2);
    }
    // First two (related to France) should both score higher P(yes) than the third
    assertThat(results.get(0)[0]).isGreaterThan(results.get(2)[0]);
    assertThat(results.get(1)[0]).isGreaterThan(results.get(2)[0]);
  }
}
