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

import static io.gravitee.llama.cpp.LlamaCppTest.EMBEDDING_MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.EMBEDDING_MODEL_TO_DOWNLOAD;
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
 * Integration tests for {@link LlamaEmbedder} using the Qwen3-Embedding-0.6B model
 * (qwen3 decoder arch, auto-detected pooling=LAST, attention=CAUSAL).
 *
 * @author Remi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class LlamaEmbedderQwen3Test extends LlamaCppTest {

  private static Arena arena;
  private static LlamaModel model;
  private static LlamaEmbedder embedder;

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

    var modelPath = getModelPath(
      EMBEDDING_MODEL_PATH,
      EMBEDDING_MODEL_TO_DOWNLOAD
    );
    model = new LlamaModel(arena, modelPath, new LlamaModelParams(arena));
    embedder = new LlamaEmbedder(
      arena,
      model,
      LlamaEmbedder.Options.defaults()
    );
  }

  @AfterAll
  static void afterAll() {
    if (embedder != null) embedder.close();
    if (model != null) model.free();
    LlamaRuntime.llama_backend_free();
    arena.close();
  }

  private static float[] l2normalize(float[] v) {
    double sum = 0;
    for (float f : v) sum += (double) f * f;
    double norm = Math.sqrt(sum);
    if (norm > 1e-9) {
      for (int i = 0; i < v.length; i++) v[i] = (float) (v[i] / norm);
    }
    return v;
  }

  private static double cosine(float[] a, float[] b) {
    double dot = 0;
    for (int i = 0; i < a.length; i++) dot += (double) a[i] * b[i];
    return dot;
  }

  @Test
  @Order(1)
  void pooling_type_should_be_auto_detected_as_last_for_qwen3() {
    assertThat(embedder.poolingType()).isEqualTo(PoolingType.LAST);
  }

  @Test
  @Order(1)
  void defaults_should_resolve_nCtx_to_model_trained_context() {
    assertThat(embedder.context().nCtx()).isEqualTo(model.nCtxTrain());
  }

  @Test
  @Order(1)
  void defaults_should_keep_llama_cpp_defaults_for_nBatch_and_nSeqMax() {
    assertThat(embedder.context().nBatch()).isEqualTo(2048);
    assertThat(embedder.context().nSeqMax()).isEqualTo(1);
  }

  @Test
  @Order(2)
  void nEmbdOut_should_match_embedding_vector_length() {
    int expected = embedder.nEmbdOut();
    assertThat(expected).isGreaterThan(0);

    float[] emb = embedder.embed("hello world");
    assertThat(emb).hasSize(expected);
  }

  @Test
  @Order(3)
  void embedding_should_be_non_zero() {
    float[] emb = embedder.embed("The quick brown fox jumps over the lazy dog");
    double norm = 0;
    for (float f : emb) norm += (double) f * f;
    assertThat(norm).isGreaterThan(0.0);
  }

  @Test
  @Order(4)
  void embedding_should_be_deterministic() {
    String text = "Determinism test sentence";
    float[] first = embedder.embed(text);
    float[] second = embedder.embed(text);
    assertThat(first).containsExactly(second);
  }

  @Test
  @Order(5)
  void similar_texts_should_have_higher_cosine_similarity_than_dissimilar() {
    float[] embA = l2normalize(
      embedder.embed("What is the capital of France?")
    );
    float[] embB = l2normalize(
      embedder.embed("Paris is the capital city of France.")
    );
    float[] embC = l2normalize(
      embedder.embed("The mitochondria is the powerhouse of the cell.")
    );

    double simAB = cosine(embA, embB);
    double simAC = cosine(embA, embC);

    assertThat(simAB)
      .as(
        "related pair (simAB=%.4f) should score higher than unrelated pair (simAC=%.4f)",
        simAB,
        simAC
      )
      .isGreaterThan(simAC);
  }

  @Test
  @Order(6)
  void embedAll_should_return_one_embedding_per_input() {
    var texts = java.util.List.of("first", "second", "third");
    var embs = embedder.embedAll(texts);
    assertThat(embs).hasSize(3);
    for (float[] emb : embs) {
      assertThat(emb).hasSize(embedder.nEmbdOut());
    }
  }
}
