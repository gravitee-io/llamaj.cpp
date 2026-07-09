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
import static io.gravitee.llama.cpp.LlamaCppTest.REASONING_MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.REASONNING_MODEL_TO_DOWNLOAD;
import static io.gravitee.llama.cpp.LlamaCppTest.getModelPath;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Validation/gating tests for the MTP (nextn) self-speculation and EAGLE3 draft sources —
 * config rejection, capability probing, head-model validation, and MTP/EAGLE3 states joining
 * a BatchIterator parallel step.
 *
 * <p>End-to-end MTP/EAGLE3 rounds require multi-GB capability models (a nextn-headed target /
 * a target-specific EAGLE3 head GGUF) that are impractical in CI; the round logic is validated
 * manually — see docs/speculative-decoding.
 *
 * @author GraviteeSource Team
 */
@Tag("integration")
class MtpEagle3SpeculativeTest extends LlamaCppTest {

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
  void staging_symbols_resolve_against_bundled_llama_cpp() {
    // The pinned llama.cpp build must expose both staging groups; if a version bump breaks the
    // mangled names, this fails fast with the per-symbol report.
    assertThat(LlamaExt.available()).as(LlamaExt.resolutionReport()).isTrue();
    assertThat(LlamaExt.eagle3Available())
      .as(LlamaExt.eagle3ResolutionReport())
      .isTrue();
  }

  @Test
  void setMtp_rejects_ngram_config() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(256).nBatch(256);
    var ctx = track(new LlamaContext(arena, model, cp));
    var state = ConversationState.create(
      arena,
      ctx,
      new LlamaTokenizer(new LlamaVocab(model), ctx),
      track(new LlamaSampler(arena).greedy())
    );

    assertThatThrownBy(() ->
      state.setMtp(ctx, SpeculativeConfig.ngramGreedy(4, 2))
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("ngram");
  }

  @Test
  void setEagle3_rejects_non_eagle3_head_model() {
    assumeTrue(LlamaExt.eagle3Available());
    Path targetPath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    Path headPath = getModelPath(
      REASONING_MODEL_PATH,
      REASONNING_MODEL_TO_DOWNLOAD
    );
    var target = track(
      new LlamaModel(arena, targetPath, new LlamaModelParams(arena))
    );
    // An ordinary LM is not an EAGLE3 head: it declares no target extract layers.
    var fakeHead = track(
      new LlamaModel(arena, headPath, new LlamaModelParams(arena))
    );
    var cp = new LlamaContextParams(arena).nCtx(256).nBatch(256);
    var ctx = track(new LlamaContext(arena, target, cp));
    var headCtx = track(new LlamaContext(arena, fakeHead, cp));
    var state = ConversationState.create(
      arena,
      ctx,
      new LlamaTokenizer(new LlamaVocab(target), ctx),
      track(new LlamaSampler(arena).greedy())
    );

    assertThatThrownBy(() ->
      state.setEagle3(headCtx, fakeHead, SpeculativeConfig.greedy(4))
    )
      .isInstanceOf(LlamaException.class)
      .hasMessageContaining("EAGLE3");
  }

  @Test
  void batchIterator_accepts_mtp_states() {
    assumeTrue(LlamaExt.available());
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(256).nBatch(256);
    var ctx = track(new LlamaContext(arena, model, cp));
    // Any same-vocab context passes setMtp's validation (the capability probe for a real nextn
    // head is native context creation; end-to-end fused rounds need a capability model and are
    // validated manually — see the class javadoc).
    var mtpish = track(new LlamaContext(arena, model, cp));
    var state = ConversationState.create(
      arena,
      ctx,
      new LlamaTokenizer(new LlamaVocab(model), ctx),
      track(new LlamaSampler(arena).greedy())
    ).setMtp(mtpish, SpeculativeConfig.greedy(4));

    assertThat(state.isMtp()).isTrue();
    assertThat(state.isSpeculative()).isTrue();

    // MTP/EAGLE3 states join the fused multi-sequence step like any other speculative state.
    try (var batchIterator = new BatchIterator(arena, ctx)) {
      batchIterator.addState(state);
    }
    state.freeSpeculativeScratch();
  }
}
