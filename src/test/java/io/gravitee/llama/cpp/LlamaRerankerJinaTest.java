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

import static io.gravitee.llama.cpp.LlamaCppTest.JINA_RERANKER_MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.JINA_RERANKER_MODEL_TO_DOWNLOAD;
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
 * Integration tests for {@link LlamaReranker} using the Jina-reranker-v1-tiny-en model
 * (bert encoder arch, auto-detected format=PLAIN, attention=NON_CAUSAL).
 *
 * <p>Jina-reranker returns {@code float[1]} with a raw relevance logit.
 *
 * @author Remi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class LlamaRerankerJinaTest extends LlamaCppTest {

  private static Arena arena;
  private static LlamaModel model;
  private static LlamaReranker reranker;

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

    var modelPath = getModelPath(
      JINA_RERANKER_MODEL_PATH,
      JINA_RERANKER_MODEL_TO_DOWNLOAD
    );
    model = new LlamaModel(arena, modelPath, new LlamaModelParams(arena));
    reranker = new LlamaReranker(
      arena,
      model,
      LlamaReranker.Options.defaults()
    );
  }

  @AfterAll
  static void afterAll() {
    if (reranker != null) reranker.close();
    if (model != null) model.free();
    LlamaRuntime.llama_backend_free();
    arena.close();
  }

  private float relevance(String query, String doc) {
    return reranker.score(query, doc)[0];
  }

  @Test
  @Order(1)
  void template_should_default_to_plain() {
    assertThat(reranker.template()).isSameAs(RerankTemplate.PLAIN);
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
  void nClsOut_should_be_one_for_jina_reranker() {
    assertThat(reranker.nClsOut()).isEqualTo(1);
  }

  @Test
  @Order(3)
  void score_should_return_single_logit() {
    float[] scores = reranker.score(
      "capital of France",
      "Paris is the capital."
    );
    assertThat(scores).hasSize(1);
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
        "relevant score (%.4f) should be higher than irrelevant score (%.4f)",
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
        "exact-match score (%.4f) should exceed unrelated score (%.4f)",
        exactScore,
        unrelatedScore
      )
      .isGreaterThan(unrelatedScore);
  }

  @Test
  @Order(7)
  void scoreAll_batched_should_match_individual_score_calls() {
    String query = "What is the capital of France?";
    var docs = java.util.List.of(
      "Paris is the capital of France.",
      "The Eiffel Tower is in Paris.",
      "Berlin is the capital of Germany.",
      "Mitochondria generate ATP.",
      "All that glitters is not gold."
    );

    var batched = reranker.scoreAll(query, docs);
    assertThat(batched).hasSize(docs.size());

    // Batched and single-call outputs are not bit-exact (packing introduces
    // tiny FP rounding differences). A tight absolute tolerance verifies
    // numerical equivalence without over-specifying the implementation.
    for (int i = 0; i < docs.size(); i++) {
      float[] single = reranker.score(query, docs.get(i));
      assertThat(batched.get(i).length).isEqualTo(single.length);
      for (int k = 0; k < single.length; k++) {
        assertThat((double) batched.get(i)[k])
          .as("batched[%d][%d] should be close to single[%d]", i, k, k)
          .isCloseTo(single[k], org.assertj.core.data.Offset.offset(0.01));
      }
    }
  }
}
