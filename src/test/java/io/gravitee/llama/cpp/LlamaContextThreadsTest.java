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

import static io.gravitee.llama.cpp.LlamaCppTest.getModelPath;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class LlamaContextThreadsTest extends LlamaCppTest {

  private static Arena arena;
  private static LlamaModel model;

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);
    model = new LlamaModel(
      arena,
      getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
      new LlamaModelParams(arena).nGpuLayers(0)
    );
  }

  @AfterAll
  static void afterAll() {
    model.free();
    LlamaRuntime.llama_backend_free();
    arena.close();
  }

  @Test
  void should_read_the_params_it_was_created_with() {
    var context = new LlamaContext(
      arena,
      model,
      new LlamaContextParams(arena).nThreads(2).nThreadsBatch(3)
    );

    assertThat(context.nThreads()).isEqualTo(2);
    assertThat(context.nThreadsBatch()).isEqualTo(3);
    context.free();
  }

  @Test
  void should_set_both_thread_counts_independently() {
    var context = new LlamaContext(arena, model, new LlamaContextParams(arena));

    context.setThreads(2, 6);

    assertThat(context.nThreads()).isEqualTo(2);
    assertThat(context.nThreadsBatch()).isEqualTo(6);
    context.free();
  }

  @Test
  void should_set_threads_on_the_context_of_a_decider() {
    var decider = LabelDecider.custom(
      arena,
      model,
      LlamaDecider.Options.defaults(),
      List.of("A", "B")
    );

    decider.context().setThreads(1, 5);

    assertThat(decider.context().nThreads()).isEqualTo(1);
    assertThat(decider.context().nThreadsBatch()).isEqualTo(5);
    decider.close();
  }
}
