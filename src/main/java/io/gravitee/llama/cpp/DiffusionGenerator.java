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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * Runs text generation for a single diffusion-model sequence (e.g. LLaDA, Dream).
 *
 * <p>Instead of autoregressive token-by-token decoding, the sequence is initialised with
 * the prompt followed by mask tokens, then refined over a fixed number of denoising steps.
 * Each step decodes the whole sequence with bidirectional attention, samples a candidate for
 * every still-masked position, and "transfers" (commits) the most confident of those
 * positions. The per-step logic lives in {@link DiffusionCanvasState}; this class owns the
 * decode loop for one canvas. For denoising many prompts in one batch, see
 * {@link BatchDiffusionIterator}.
 *
 * <p><b>Not yet supported</b> (silently ignored / unavailable): classifier-free guidance
 * (CFG) and the stochastic transfer path ({@code algTemp > 0}).
 *
 * <p>RNG note: gumbel noise and the {@code RANDOM} algorithm use {@link java.util.Random},
 * so outputs are reproducible for a given seed within this library but will not bit-match
 * the C++ {@code std::mt19937} reference at non-zero temperature.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class DiffusionGenerator {

  /** Optional per-step progress hook. Return {@code false} to stop early. */
  @FunctionalInterface
  public interface StepCallback {
    boolean onStep(int step, int totalSteps, int[] tokens);
  }

  // Single-sequence diffusion always uses sequence id 0; cached to avoid a per-token
  // List.of(0) allocation in the decode loop.
  private static final List<Integer> SEQ_0 = List.of(0);

  private final LlamaContext context;

  public DiffusionGenerator(LlamaContext context) {
    this.context = context;
  }

  /**
   * Generates a full sequence by denoising.
   *
   * @param promptTokens The input prompt token ids
   * @param params       Generation configuration ({@code maxLength} is required and must
   *                     exceed the prompt length; the mask token is resolved from the model
   *                     when {@code maskTokenId} is unset)
   * @return The decoded sequence of length {@code params.maxLength()}, prompt included
   */
  public int[] generate(int[] promptTokens, DiffusionParams params) {
    return generate(promptTokens, params, null);
  }

  public int[] generate(
    int[] promptTokens,
    DiffusionParams params,
    StepCallback callback
  ) {
    if (!context.getModel().isDiffusion()) {
      throw new LlamaException("Model is not a diffusion model");
    }
    int maskToken = DiffusionCanvasState.resolveMaskToken(context, params);
    boolean shiftLogits = params.shiftLogits() != null
      ? params.shiftLogits()
      : DiffusionCanvasState.resolveShiftLogits(context);
    int nVocab = context.nVocab();
    var vocab = new LlamaVocab(context.getModel());
    var canvas = new DiffusionCanvasState(
      0,
      promptTokens,
      maskToken,
      shiftLogits,
      params
    );

    // Bidirectional attention for the duration of generation.
    LlamaRuntime.llama_set_causal_attn(context.segment, false);
    LlamaSampler sampler = null;
    LlamaBatch batch = null;
    // Managed manually (not try-with-resources): the batch/sampler structs live in this
    // arena, so they must be freed BEFORE the arena closes — a try-with-resources would
    // close the arena first, leaving llama_batch_free reading freed memory.
    Arena arena = Arena.ofConfined();
    try {
      sampler = DiffusionSamplers.build(arena, params);
      var candidates = new LlamaTokenDataArray(arena, nVocab);
      batch = new LlamaBatch(arena, canvas.maxLength(), 0, 1);
      batch.enableCache();

      for (int blockNum = 0; blockNum < canvas.numBlocks(); blockNum++) {
        canvas.beginBlock(blockNum);
        for (int step = 0; step < canvas.stepsPerBlock(); step++) {
          int globalStep = blockNum * canvas.stepsPerBlock() + step;
          if (
            callback != null &&
            !callback.onStep(globalStep, params.steps(), canvas.tokens())
          ) {
            return canvas.tokens();
          }
          if (canvas.done()) {
            break;
          }
          int decodeLen = canvas.decodeLength(blockNum);
          MemorySegment logits = decodeStep(
            batch,
            canvas.tokens(),
            decodeLen,
            nVocab
          );
          canvas.applyStep(
            blockNum,
            step,
            logits,
            0,
            nVocab,
            candidates,
            sampler
          );
          // Stop once the answer is settled (committed prefix ending in EOS).
          if (canvas.answerComplete(vocab::isEog)) {
            return canvas.tokens();
          }
        }
      }
      return canvas.tokens();
    } finally {
      // Free native-owned resources (the malloc'd batch arrays / sampler chain) while the
      // arena is still open, then close the arena, then restore causal attention.
      if (batch != null) {
        batch.free();
      }
      if (sampler != null) {
        sampler.free();
      }
      arena.close();
      LlamaRuntime.llama_set_causal_attn(context.segment, true);
    }
  }

  private MemorySegment decodeStep(
    LlamaBatch batch,
    int[] tokens,
    int decodeLen,
    int nVocab
  ) {
    batch.clear();
    // Only decode the live prefix (whole canvas for timestep, current block for block mode).
    for (int i = 0; i < decodeLen; i++) {
      batch.add(tokens[i], i, SEQ_0, true);
    }
    int ret = context.decode(batch);
    if (ret != 0) {
      throw new LlamaException("Diffusion decode failed with code " + ret);
    }
    MemorySegment logits = LlamaRuntime.llama_get_logits(context.segment);
    if (logits == null || logits.address() == 0) {
      throw new LlamaException(
        "llama_get_logits returned NULL during diffusion step"
      );
    }
    return logits.reinterpret(
      (long) decodeLen * nVocab * ValueLayout.JAVA_FLOAT.byteSize()
    );
  }
}
