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

import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_TO_DOWNLOAD;
import static io.gravitee.llama.cpp.LlamaCppTest.getModelPath;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class GraphReallocTest extends LlamaCppTest {

  private static final int MAX_TOKENS = 32;

  private static Arena arena;

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);
  }

  @AfterAll
  static void afterAll() {
    LlamaRuntime.llama_backend_free();
    arena.close();
    arena = null;
  }

  @Test
  void steady_single_token_decoding_does_not_reallocate_the_graph() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var ctx = track(
      new LlamaContext(
        arena,
        model,
        new LlamaContextParams(arena).nCtx(512).noPerf(false)
      )
    );
    var tokenizer = new LlamaTokenizer(new LlamaVocab(model), ctx);
    var state = ConversationState.create(
      arena,
      ctx,
      tokenizer,
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .initialize("Count from one to one hundred: one, two, three,");
    var it = new DefaultLlamaIterator(state);

    assertThat(it.hasNext()).isTrue();
    it.next();
    var afterFirst = ctx.getPerformance(arena);
    int generated = 1;
    while (it.hasNext()) {
      it.next();
      generated++;
    }
    var afterAll = ctx.getPerformance(arena);

    assertThat(generated).isGreaterThan(8);
    assertThat(afterAll.graphReallocsSameSize()).isGreaterThanOrEqualTo(0);
    assertThat(afterAll.graphReallocsResized()).isGreaterThanOrEqualTo(0);
    assertThat(afterAll.graphReallocs() - afterFirst.graphReallocs())
      .as("reallocations while decoding %d single tokens", generated - 1)
      .isLessThanOrEqualTo(2);
  }
}
