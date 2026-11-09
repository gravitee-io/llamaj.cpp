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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.gravitee.llama.cpp.LlamaModelParams.TensorOverride;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests for tensor buffer overrides and the mixture-of-experts device cache. The expert case runs
 * only when {@code LLAMAJ_MOE_MODEL} points at a mixture-of-experts GGUF.
 */
class MoeOffloadTest extends LlamaCppTest {

  private static final String PROMPT =
    "Count from one to twenty in words, separated by commas: one, two,";

  private static Arena arena;
  private static LlamaLogger logger;
  private static final List<String> logs = new CopyOnWriteArrayList<>();

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);
    logger = new LlamaLogger(Arena.global());
    logger.setLogging(LlamaLogLevel.DEBUG, logs::add);
  }

  @AfterEach
  void clearLogs() {
    logs.clear();
  }

  @AfterAll
  static void afterAll() {
    logger.setLogging(LlamaLogLevel.ERROR);
    LlamaRuntime.llama_backend_free();
    arena.close();
    arena = null;
  }

  @Test
  void cpu_moe_overrides_the_expert_tensors_of_each_layer() {
    var params = new LlamaModelParams(arena).cpuMoe(arena, 2);
    assertThat(params.tensorOverrides())
      .extracting(TensorOverride::pattern)
      .containsExactly(
        "blk\\.0" + LlamaModelParams.EXPERT_TENSORS,
        "blk\\.1" + LlamaModelParams.EXPERT_TENSORS
      );
    assertThat(
      BackendRegistry.bufferTypeName(
        params.tensorOverrides().getFirst().bufferType()
      )
    ).isEqualTo("CPU");

    params.cpuMoe(arena);
    assertThat(params.tensorOverrides())
      .last()
      .extracting(TensorOverride::pattern)
      .isEqualTo(LlamaModelParams.EXPERT_TENSORS);

    var cpu = BackendRegistry.cpuBufferType();
    assertThatThrownBy(() -> new TensorOverride("", cpu)).isInstanceOf(
      IllegalArgumentException.class
    );
    assertThatThrownBy(() ->
      new LlamaContextParams(arena).moeCacheSize(-1)
    ).isInstanceOf(IllegalArgumentException.class);
    assertThat(
      new LlamaContextParams(arena).moeCacheSize(64L << 20).moeCacheSize()
    ).isEqualTo(64L << 20);
  }

  @Test
  void a_tensor_override_moves_the_matching_weights_and_keeps_the_output() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    String plain = generate(path, new LlamaModelParams(arena), 0);

    logs.clear();
    var params = new LlamaModelParams(arena).tensorOverrides(
      arena,
      List.of(
        new TensorOverride("blk\\.0\\.ffn_up", BackendRegistry.cpuBufferType())
      )
    );
    String moved = generate(path, params, 0);

    assertThat(logs).anyMatch(
      l ->
        l.contains("blk.0.ffn_up.weight") &&
        l.contains("buffer type overridden to CPU")
    );
    assertThat(logs).noneMatch(
      l -> l.contains("blk.1.ffn_up.weight") && l.contains("overridden")
    );
    assertThat(moved).isEqualTo(plain);
  }

  @Test
  void a_device_cache_needs_a_mixture_of_experts_model() {
    var model = track(
      new LlamaModel(
        arena,
        getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
        new LlamaModelParams(arena)
      )
    );
    assertThatThrownBy(() ->
      new LlamaContext(
        arena,
        model,
        new LlamaContextParams(arena).nCtx(256).moeCacheSize(16L << 20)
      )
    ).isInstanceOf(LlamaException.class);
    assertThat(logs).anyMatch(l ->
      l.contains("MoE cache requires a MoE model")
    );
  }

  @Test
  void experts_in_host_memory_with_a_device_cache_decode_the_same() {
    String env = System.getenv("LLAMAJ_MOE_MODEL");
    assumeTrue(
      env != null && Files.exists(Path.of(env)),
      "LLAMAJ_MOE_MODEL not set"
    );
    Path path = Path.of(env);
    String plain = generate(path, new LlamaModelParams(arena), 0);

    logs.clear();
    String offloaded = generate(
      path,
      new LlamaModelParams(arena).useExtraBufferTypes(false).cpuMoe(arena),
      256L << 20
    );
    System.out.println("plain     : " + plain);
    System.out.println("offloaded : " + offloaded);
    logs
      .stream()
      .filter(l -> l.contains("MoE cache") || l.contains("llama_moe_cache"))
      .forEach(System.out::print);

    assertThat(logs).anyMatch(
      l ->
        l.contains("_exps.weight") &&
        l.contains("buffer type overridden to CPU")
    );
    assertThat(logs).anyMatch(l -> l.contains("MoE cache size"));
    assertThat(offloaded).isEqualTo(plain);
  }

  private String generate(Path path, LlamaModelParams params, long cacheBytes) {
    var model = track(new LlamaModel(arena, path, params));
    var ctx = track(
      new LlamaContext(
        arena,
        model,
        new LlamaContextParams(arena)
          .nCtx(512)
          .nBatch(512)
          .moeCacheSize(cacheBytes)
      )
    );
    var tokenizer = new LlamaTokenizer(new LlamaVocab(model), ctx);
    var state = ConversationState.create(
      arena,
      ctx,
      tokenizer,
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(32)
      .initialize(PROMPT);
    return new DefaultLlamaIterator(state)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", String::concat);
  }
}
