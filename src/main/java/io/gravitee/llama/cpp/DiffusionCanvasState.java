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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Random;
import java.util.function.IntPredicate;

import static java.util.Optional.ofNullable;

/**
 * Mutable per-canvas state and step logic for diffusion generation, shared by the
 * single-sequence {@link DiffusionGenerator} and the multi-sequence
 * {@link BatchDiffusionIterator}.
 *
 * <p>A "canvas" is one sequence being denoised: the prompt followed by mask tokens,
 * refined in place across denoising steps. The decode itself is driven externally
 * (so a batch can decode many canvases at once); this class owns everything else —
 * mask tracking, the transfer schedule, sampling each masked position through the
 * sampler chain, confidence scoring, and committing the most confident positions.
 *
 * <p>Block boundaries derive from this canvas's own prompt length, so canvases with
 * different prompt lengths can share one batched run.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
final class DiffusionCanvasState {

  private final int seqId;
  private final int[] tokens;
  private final int maskToken;
  private final int maxLength;
  private final int nInput;
  private final DiffusionParams params;
  private final boolean shiftLogits;
  private final boolean blockBased;
  private final int numBlocks;
  private final int stepsPerBlock;
  private final Random rng;

  // Recomputed at the start of each block (block-based schedule only).
  private int[] numTransferTokens;

  // Per-step scratch, allocated once and reused to avoid hot-path garbage.
  private final int[] maskPosScratch;
  private final float[] confScratch;
  private final int[] sampledScratch;
  private final boolean[] takenScratch;
  private final int[] committed; // canvas positions committed by the last applyStep
  private int nCommitted;

  DiffusionCanvasState(
    int seqId,
    int[] promptTokens,
    int maskToken,
    boolean shiftLogits,
    DiffusionParams params
  ) {
    this.seqId = seqId;
    this.maskToken = maskToken;
    this.shiftLogits = shiftLogits;
    this.params = params;
    this.maxLength = params.maxLength();
    this.nInput = promptTokens.length;
    if (nInput <= 0 || maxLength <= nInput) {
      throw new LlamaException(
        "maxLength (" +
          maxLength +
          ") must exceed prompt length (" +
          nInput +
          ")"
      );
    }
    this.tokens = new int[maxLength];
    System.arraycopy(promptTokens, 0, tokens, 0, nInput);
    Arrays.fill(tokens, nInput, maxLength, maskToken);

    this.blockBased =
      params.schedule() == DiffusionTransferSchedule.BLOCK_BASED;
    if (blockBased) {
      if (params.blockLength() <= 0 || maxLength % params.blockLength() != 0) {
        throw new LlamaException(
          "BLOCK_BASED schedule requires blockLength to divide maxLength"
        );
      }
      this.numBlocks = maxLength / params.blockLength();
      if (params.steps() % numBlocks != 0) {
        throw new LlamaException(
          "BLOCK_BASED schedule requires steps to be a multiple of the block count"
        );
      }
      this.stepsPerBlock = params.steps() / numBlocks;
    } else {
      this.numBlocks = 1;
      this.stepsPerBlock = params.steps();
    }
    // Seed per canvas so batched runs stay deterministic and independent of order.
    this.rng = new Random(params.seed() + seqId);

    this.maskPosScratch = new int[maxLength];
    this.confScratch = new float[maxLength];
    this.sampledScratch = new int[maxLength];
    this.takenScratch = new boolean[maxLength];
    this.committed = new int[maxLength];
  }

  int seqId() {
    return seqId;
  }

  int[] tokens() {
    return tokens;
  }

  int maxLength() {
    return maxLength;
  }

  int numBlocks() {
    return numBlocks;
  }

  /**
   * Number of leading positions that need to be decoded for the given block. Under the
   * block schedule only the prompt plus blocks up to and including the current one are
   * live — future blocks are still fully masked and must not be attended to (this matches
   * LLaDA's semi-autoregressive masking and avoids decoding the whole canvas every step).
   * The timestep schedule denoises the entire canvas, so it returns {@code maxLength}.
   */
  int decodeLength(int blockNum) {
    return blockBased ? blockEnd(blockNum) : maxLength;
  }

  int stepsPerBlock() {
    return stepsPerBlock;
  }

  /** {@code true} once every position has been unmasked. */
  boolean done() {
    for (int token : tokens) {
      if (token == maskToken) {
        return false;
      }
    }
    return true;
  }

  /**
   * {@code true} once the answer is settled: the generated region is committed (no masks)
   * up to a committed end-of-generation token. Everything after it is padding, so denoising
   * can stop early. Because committed positions are never re-sampled, a committed prefix
   * ending in EOS is final — safe to stop even though diffusion fills out of order.
   *
   * @param isEog Predicate identifying end-of-generation tokens (e.g. {@code vocab::isEog})
   */
  boolean answerComplete(IntPredicate isEog) {
    for (int i = nInput; i < maxLength; i++) {
      if (tokens[i] == maskToken) {
        return false; // a hole before any terminator — not settled yet
      }
      if (isEog.test(tokens[i])) {
        return true; // committed EOS with a fully-committed prefix
      }
    }
    return false;
  }

  private int blockStart(int blockNum) {
    return blockBased ? nInput + blockNum * params.blockLength() : 0;
  }

  private int blockEnd(int blockNum) {
    return blockBased
      ? Math.min(nInput + (blockNum + 1) * params.blockLength(), maxLength)
      : maxLength;
  }

  /** Called once at the start of each block to set up block-based transfer counts. */
  void beginBlock(int blockNum) {
    if (!blockBased) {
      return;
    }
    int blockMaskCount = 0;
    for (int i = blockStart(blockNum); i < blockEnd(blockNum); i++) {
      if (tokens[i] == maskToken) {
        blockMaskCount++;
      }
    }
    numTransferTokens = transferCounts(blockMaskCount, stepsPerBlock);
  }

  /**
   * Samples each masked position from the freshly decoded logits and commits the
   * most confident ones, mutating {@link #tokens} in place.
   *
   * @param blockNum       Current block index
   * @param step           Step index within the block
   * @param logits         Decoded logits buffer covering this canvas's rows
   * @param canvasRowBase  Row (output-token) index where this canvas's logits begin
   *                       in {@code logits}; {@code 0} for a single-canvas decode
   * @param nVocab         Vocabulary size (row stride, in floats)
   * @param candidates     Reusable candidate buffer
   * @param sampler        The sampler chain
   */
  void applyStep(
    int blockNum,
    int step,
    MemorySegment logits,
    long canvasRowBase,
    int nVocab,
    LlamaTokenDataArray candidates,
    LlamaSampler sampler
  ) {
    nCommitted = 0;
    int start = blockStart(blockNum);
    int end = blockEnd(blockNum);

    int m = 0;
    for (int i = 0; i < maxLength; i++) {
      boolean inBlock = !blockBased || (i >= start && i < end);
      if (tokens[i] == maskToken && inBlock) {
        maskPosScratch[m++] = i;
      }
    }
    if (m == 0) {
      return;
    }

    if (params.addGumbelNoise() && params.temperature() > 0.0f) {
      applyGumbelNoise(
        logits,
        canvasRowBase * nVocab,
        (long) maxLength * nVocab,
        params.temperature(),
        rng
      );
    }

    int transferCount = transferCount(
      step,
      stepsPerBlock,
      m,
      params,
      numTransferTokens
    );

    // Sample a candidate + confidence for each masked position in this block.
    for (int k = 0; k < m; k++) {
      long row = canvasRowBase + logitRow(maskPosScratch[k], shiftLogits);
      candidates.fill(logits, row * nVocab);
      candidates.apply(sampler);
      sampledScratch[k] = candidates.selectedId();
      confScratch[k] = confidence(candidates, params.algorithm());
    }

    if (params.algorithm() == DiffusionAlgorithm.ORIGIN) {
      float pTransfer = (float) transferCount / m;
      for (int k = 0; k < m; k++) {
        if (rng.nextFloat() < pTransfer) {
          commit(maskPosScratch[k], sampledScratch[k]);
        }
      }
    } else if (transferCount > 0) {
      // Commit the `transferCount` most-confident positions via top-N selection
      // (ties → lower position index, matching the reference), no boxing/sort.
      int toCommit = Math.min(transferCount, m);
      Arrays.fill(takenScratch, 0, m, false);
      for (int c = 0; c < toCommit; c++) {
        int best = -1;
        for (int k = 0; k < m; k++) {
          if (!takenScratch[k] && (best == -1 || confScratch[k] > confScratch[best])) {
            best = k;
          }
        }
        takenScratch[best] = true;
        commit(maskPosScratch[best], sampledScratch[best]);
      }
    }
  }

  private void commit(int pos, int token) {
    tokens[pos] = token;
    committed[nCommitted++] = pos;
  }

  /**
   * Canvas positions committed by the most recent {@link #applyStep} — only the first
   * {@link #committedCount()} entries are valid. The backing array is reused, so read it
   * before the next {@code applyStep}.
   */
  int[] committedPositions() {
    return committed;
  }

  /** Number of positions committed by the most recent {@link #applyStep}. */
  int committedCount() {
    return nCommitted;
  }

  /* ----- shared pure helpers ----- */

  private static long logitRow(int pos, boolean shiftLogits) {
    if (shiftLogits) {
      return pos == 0 ? 0 : pos - 1;
    }
    return pos;
  }

  static int transferCount(
    int step,
    int totalSteps,
    int remainingMasked,
    DiffusionParams params,
    int[] numTransferTokens
  ) {
    return switch (params.schedule()) {
      case TIMESTEP_BASED -> timestep(step, totalSteps, remainingMasked, params);
      case BLOCK_BASED -> ofNullable(numTransferTokens)
              .filter(tt -> step < numTransferTokens.length)
              .map(tt -> tt[step])
              .orElse(remainingMasked / (totalSteps - step));
    };
  }

  private static int timestep(int step, int totalSteps, int remainingMasked, DiffusionParams params) {
    float eps = params.eps();
    float t = 1.0f - ((float) step / totalSteps) * (1.0f - eps);
    float s = 1.0f - ((float) (step + 1) / totalSteps) * (1.0f - eps);
    float pTransfer = (step < totalSteps - 1) ? (1.0f - s / t) : 1.0f;
    return (int) (remainingMasked * pTransfer);
  }

  static int[] transferCounts(int maskCount, int steps) {
    int[] counts = new int[steps];
    int base = maskCount / steps;
    int remainder = maskCount % steps;
    for (int i = 0; i < steps; i++) {
      counts[i] = base + (i < remainder ? 1 : 0);
    }
    return counts;
  }

  private float confidence(LlamaTokenDataArray candidates, DiffusionAlgorithm algorithm) {
    return switch (algorithm) {
      case CONFIDENCE_BASED, ORIGIN -> candidates.selectedProbability();
      case ENTROPY_BASED -> entropy(candidates);
      case MARGIN_BASED -> margin(candidates);
      case RANDOM ->  rng.nextFloat();
    };
  }

  private static float margin(LlamaTokenDataArray candidates) {
    float top1 = 0.0f;
    float top2 = 0.0f;
    long n = candidates.size();
    for (long i = 0; i < n; i++) {
      float prob = candidates.probabilityAt(i);
      if (prob > top1) {
        top2 = top1;
        top1 = prob;
      } else if (prob > top2) {
        top2 = prob;
      }
    }
    return top1 - top2;
  }

  private static float entropy(LlamaTokenDataArray candidates) {
    float entropy = 0.0f;
    float epsilon = 1e-10f;
    long n = candidates.size();
    for (long i = 0; i < n; i++) {
      float prob = candidates.probabilityAt(i);
      entropy += prob * (float) Math.log(prob + epsilon);
    }
    return -entropy;
  }

  private static void applyGumbelNoise(
    MemorySegment logits,
    long startFloat,
    long count,
    float temperature,
    Random rng
  ) {
    long end = startFloat + count;
    for (long i = startFloat; i < end; i++) {
      double noise = Math.max(rng.nextDouble(), 1e-20);
      double gumbel = Math.pow(-Math.log(noise), temperature);
      float l = logits.getAtIndex(ValueLayout.JAVA_FLOAT, i);
      logits.setAtIndex(
        ValueLayout.JAVA_FLOAT,
        i,
        (float) (Math.exp(l) / gumbel)
      );
    }
  }

  /**
   * Resolves the mask token from the model when the caller hasn't set one explicitly.
   *
   * @throws LlamaException if neither the params nor the model define a mask token
   */
  static int resolveMaskToken(LlamaContext context, DiffusionParams params) {
    int maskToken = params.maskTokenId();
    if (maskToken == DiffusionParams.LLAMA_TOKEN_NULL) {
      maskToken = LlamaRuntime.llama_vocab_mask(
        LlamaRuntime.llama_model_get_vocab(context.getModel().segment)
      );
    }
    if (maskToken == DiffusionParams.LLAMA_TOKEN_NULL) {
      throw new LlamaException(
        "Model defines no mask token; set DiffusionParams.maskTokenId() explicitly"
      );
    }
    return maskToken;
  }

  /**
   * Resolves whether logits are shifted by one position, from the GGUF metadata key
   * {@code diffusion.shift_logits}. This is an architectural property of the model, not a
   * tuning knob: Dream-style models predict the next token (shift = true), while LLaDA-style
   * models predict the token at each masked position (shift = false). Read from the GGUF
   * {@code diffusion.shift_logits} metadata, defaulting to {@code true} when the key is
   * absent (matching llama.cpp's diffusion CLI). Callers can override via
   * {@link DiffusionParams#shiftLogits(Boolean)} when a GGUF omits or misdeclares the key.
   */
  static boolean resolveShiftLogits(LlamaContext context) {
    String value = context
      .getModel()
      .metaVal(context.getArena(), "diffusion.shift_logits");
    return value == null || Boolean.parseBoolean(value);
  }
}
