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

import io.gravitee.llama.cpp.LlamaTokenizer.TokenizerResponse;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * End-to-end test for {@link DiffusionGenerator} against a real diffusion GGUF.
 *
 * <p>Gated behind the {@code RUN_DIFFUSION_TEST=true} environment variable because it
 * downloads a multi-GB model (LLaDA-8B-Instruct, ~5 GB at Q4_K_M) — far larger than the
 * sub-GB models the rest of the suite uses — and a few denoising steps are comparatively
 * heavy. Run locally on Apple Silicon with:
 * <pre>{@code
 * RUN_DIFFUSION_TEST=true mvn -P macosx-aarch64 test -Dtest=DiffusionGeneratorTest
 * }</pre>
 *
 * <p>Native resources are freed via {@link #track(Freeable)} per the Metal teardown
 * contract in {@link LlamaCppTest}.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@EnabledIfEnvironmentVariable(named = "RUN_DIFFUSION_TEST", matches = "true")
class DiffusionGeneratorTest extends LlamaCppTest {

  static final String DIFFUSION_MODEL_TO_DOWNLOAD =
    "https://huggingface.co/mradermacher/LLaDA-8B-Instruct-GGUF/resolve/main/LLaDA-8B-Instruct.Q4_K_M.gguf";
  static final String DIFFUSION_MODEL_PATH = "models/diffusion.gguf";

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
  void model_should_be_detected_as_diffusion() {
    Path path = getModelPath(DIFFUSION_MODEL_PATH, DIFFUSION_MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    assertThat(model.isDiffusion()).isTrue();

    var vocab = new LlamaVocab(model);
    assertThat(vocab.maskToken())
      .as("diffusion model must define a mask token")
      .isNotEqualTo(DiffusionParams.LLAMA_TOKEN_NULL);
  }

  @Test
  void generate_should_unmask_all_positions_and_produce_text() {
    Path path = getModelPath(DIFFUSION_MODEL_PATH, DIFFUSION_MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));

    int maxLength = 64;
    // The whole sequence is decoded in a single (u)batch per step, so both must fit it.
    var contextParams = new LlamaContextParams(arena)
      .nCtx(maxLength)
      .nBatch(maxLength)
      .nUBatch(maxLength);
    var context = track(new LlamaContext(arena, model, contextParams));
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);

    int[] prompt = toTokens(
      tokenizer.tokenize(arena, "The capital of France is")
    );
    assertThat(prompt.length).isLessThan(maxLength);

    var params = new DiffusionParams()
      .maxLength(maxLength)
      .steps(32)
      .algorithm(DiffusionAlgorithm.CONFIDENCE_BASED)
      .schedule(DiffusionTransferSchedule.TIMESTEP_BASED)
      .seed(42);

    int[] output = new DiffusionGenerator(context).generate(prompt, params);

    assertThat(output).hasSize(maxLength);
    int maskToken = vocab.maskToken();
    for (int i = 0; i < output.length; i++) {
      assertThat(output[i])
        .as("position %d should have been unmasked", i)
        .isNotEqualTo(maskToken);
    }

    String text = detokenize(vocab, output);
    System.out.println("Diffusion output: " + text);
    assertThat(text).isNotBlank();
  }

  @Test
  void block_schedule_should_also_generate() {
    Path path = getModelPath(DIFFUSION_MODEL_PATH, DIFFUSION_MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));

    int maxLength = 64;
    var contextParams = new LlamaContextParams(arena)
      .nCtx(maxLength)
      .nBatch(maxLength)
      .nUBatch(maxLength);
    var context = track(new LlamaContext(arena, model, contextParams));
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);

    int[] prompt = toTokens(tokenizer.tokenize(arena, "Once upon a time"));

    var params = new DiffusionParams()
      .maxLength(maxLength)
      .steps(32)
      .schedule(DiffusionTransferSchedule.BLOCK_BASED)
      .blockLength(32) // must divide maxLength; step count must be a multiple of block count
      .seed(7);

    int[] output = new DiffusionGenerator(context).generate(prompt, params);
    assertThat(output).hasSize(maxLength);
    assertThat(detokenize(vocab, output)).isNotBlank();
  }

  @Test
  void iterator_should_stream_positioned_tokens_for_all_sequences() {
    Path path = getModelPath(DIFFUSION_MODEL_PATH, DIFFUSION_MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));

    int maxLength = 48;
    int nSeq = 2;
    var contextParams = new LlamaContextParams(arena)
      .nCtx(nSeq * maxLength)
      .nBatch(nSeq * maxLength)
      .nUBatch(nSeq * maxLength);
    var context = track(new LlamaContext(arena, model, contextParams));
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);

    var params = new DiffusionParams().maxLength(maxLength).steps(32).seed(5);

    var it = new BatchDiffusionIterator(arena, context, params);
    try {
      it.addState(
        0,
        toTokens(tokenizer.tokenize(arena, "The capital of France is"))
      );
      it.addState(1, toTokens(tokenizer.tokenize(arena, "Water boils at")));

      Map<Integer, String[]> canvases = new HashMap<>();
      canvases.put(0, new String[maxLength]);
      canvases.put(1, new String[maxLength]);
      Set<Integer> finals = new HashSet<>();
      int positioned = 0;

      while (it.hasNext()) {
        DiffusionToken token = it.next();
        if (token.isFinal()) {
          assertThat(token.finishReason()).isNotNull();
          assertThat(token.position()).isEqualTo(-1);
          finals.add(token.seqId());
        } else {
          assertThat(token.position()).isBetween(0, maxLength - 1);
          // Each position is finalized exactly once.
          assertThat(canvases.get(token.seqId())[token.position()]).isNull();
          canvases.get(token.seqId())[token.position()] = token.text();
          positioned++;
        }
      }

      assertThat(finals).containsExactlyInAnyOrder(0, 1);
      assertThat(positioned).isGreaterThan(0);
      assertThat(it.hasActiveSequences()).isFalse();

      // Reconstruct each sequence by position; the generated region must be non-empty.
      canvases.forEach((seqId, slots) -> {
        var sb = new StringBuilder();
        for (String piece : slots) {
          if (piece != null) {
            sb.append(piece);
          }
        }
        System.out.println("Iterator seq " + seqId + ": " + sb);
        assertThat(sb.toString()).as("seq %d text", seqId).isNotBlank();
      });
    } finally {
      it.free();
    }
  }

  @Test
  void iterator_removeState_should_cancel_one_sequence() {
    Path path = getModelPath(DIFFUSION_MODEL_PATH, DIFFUSION_MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));

    int maxLength = 48;
    int nSeq = 2;
    var contextParams = new LlamaContextParams(arena)
      .nCtx(nSeq * maxLength)
      .nBatch(nSeq * maxLength)
      .nUBatch(nSeq * maxLength);
    var context = track(new LlamaContext(arena, model, contextParams));
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);

    var params = new DiffusionParams().maxLength(maxLength).steps(32).seed(9);

    var it = new BatchDiffusionIterator(arena, context, params);
    try {
      it.addState(
        0,
        toTokens(tokenizer.tokenize(arena, "The capital of France is"))
      );
      it.addState(1, toTokens(tokenizer.tokenize(arena, "Water boils at")));

      // Advance at least one step, then cancel sequence 1.
      assertThat(it.hasNext()).isTrue();
      it.next();
      assertThat(it.removeState(1)).isTrue();
      assertThat(it.removeState(1)).as("already removed").isFalse();

      Set<Integer> finals = new HashSet<>();
      while (it.hasNext()) {
        DiffusionToken token = it.next();
        assertThat(token.seqId())
          .as("no tokens for cancelled sequence after removal")
          .isNotEqualTo(1);
        if (token.isFinal()) {
          finals.add(token.seqId());
        }
      }

      // Only the survivor finalizes; the cancelled sequence emits no final marker.
      assertThat(finals).containsExactly(0);
      assertThat(it.hasActiveSequences()).isFalse();
    } finally {
      it.free();
    }
  }

  private static int[] toTokens(TokenizerResponse response) {
    int[] tokens = new int[response.size()];
    for (int i = 0; i < response.size(); i++) {
      tokens[i] = response.data().getAtIndex(ValueLayout.JAVA_INT, i);
    }
    return tokens;
  }

  private static String detokenize(LlamaVocab vocab, int[] tokens) {
    var sb = new StringBuilder();
    for (int token : tokens) {
      sb.append(new String(vocab.tokenToPiece(token), StandardCharsets.UTF_8));
    }
    return sb.toString();
  }
}
