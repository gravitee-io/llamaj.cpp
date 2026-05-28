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

import static io.gravitee.llama.cpp.LlamaRuntime.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaContext extends MemorySegmentAware implements Freeable {

  private final Arena arena;
  private final LlamaModel model;
  private final int nCtx;
  private final int nBatch;
  private final int nUBatch;
  private final int nSeqMax;
  private final LlamaMemory memory;

  public LlamaContext(
    Arena arena,
    LlamaModel model,
    LlamaContextParams params
  ) {
    super(llama_init_from_model(model.segment, params.segment));
    if (segment == null || segment.address() == 0) {
      throw new LlamaException("Failed to create context from model");
    }
    this.arena = arena;
    this.model = model;
    this.nCtx = LlamaRuntime.llama_n_ctx(segment);
    this.nBatch = LlamaRuntime.llama_n_batch(segment);
    this.nUBatch = LlamaRuntime.llama_n_ubatch(segment);
    this.nSeqMax = LlamaRuntime.llama_n_seq_max(segment);
    this.memory = new LlamaMemory(this);
  }

  public Arena getArena() {
    return arena;
  }

  public LlamaModel getModel() {
    return model;
  }

  public int nCtx() {
    return nCtx;
  }

  public int nBatch() {
    return nBatch;
  }

  public int nUBatch() {
    return nUBatch;
  }

  public int nSeqMax() {
    return nSeqMax;
  }

  public int nVocab() {
    return llama_n_vocab(llama_model_get_vocab(model.segment));
  }

  public int nCtxUsedCells() {
    return memory.posMax() - memory.posMin() + 1;
  }

  public void clearCache() {
    checkNotFreed();
    memory.clear();
  }

  public LlamaMemory getMemory() {
    checkNotFreed();
    return memory;
  }

  public int decode(LlamaBatch batch) {
    checkNotFreed();
    return llama_decode(segment, batch.segment);
  }

  /**
   * Enables or disables embedding extraction on this context at runtime.
   * Equivalent to the {@code embeddings} flag in {@link LlamaContextParams} but settable
   * after context creation.
   *
   * @param embeddings {@code true} to activate embedding output
   */
  public void setEmbeddings(boolean embeddings) {
    checkNotFreed();
    llama_set_embeddings(segment, embeddings);
  }

  /**
   * Returns the pooled embedding vector for the given sequence.
   * <p>
   * Use when the context was created with a pooling type other than {@code NONE}
   * (e.g. {@link PoolingType#MEAN}, {@link PoolingType#CLS}, {@link PoolingType#LAST}).
   * When {@link PoolingType#RANK} is active the returned array contains
   * {@code nClsOut} reranker / classifier scores instead of a full embedding vector.
   * The size is determined by the active pooling type:
   * <ul>
   *   <li>{@link PoolingType#RANK}: {@code float[nClsOut]} (per-class relevance scores)</li>
   *   <li>Any other non-NONE pooling: {@code float[nEmbdOut]} (pooled embedding vector)</li>
   * </ul>
   * <p>
   * Returns {@code null} when pooling is {@code NONE}.
   * The context must have been created with {@code embeddings=true}.
   *
   * @param seqId The sequence id assigned when building the batch
   * @return A new {@code float[]} copy of the pooled embedding / score vector,
   *         or {@code null} if pooling is disabled
   */
  public float[] getEmbeddingsSeq(int seqId) {
    checkNotFreed();
    MemorySegment ptr = llama_get_embeddings_seq(segment, seqId);
    if (ptr == null || ptr.address() == 0) {
      return null;
    }
    // For RANK pooling the buffer holds n_cls_out floats (classifier scores).
    // For all other non-NONE pooling types it holds n_embd_out floats.
    PoolingType pooling = llama_pooling_type(segment);
    int size = pooling == PoolingType.RANK ? model.nClsOut() : model.nEmbdOut();
    return ptr
      .reinterpret(size * ValueLayout.JAVA_FLOAT.byteSize())
      .toArray(ValueLayout.JAVA_FLOAT);
  }

  /**
   * Returns the embedding vector for the i-th token in the last decoded batch.
   * <p>
   * Use when the context was created with {@link PoolingType#NONE} and you need
   * per-token embeddings (e.g. for span-based NER heads). Negative indices count from
   * the end ({@code -1} = last token in the batch).
   * <p>
   * The token at index {@code i} must have been added to the batch with
   * {@code logits=true}, and the context must have been created with
   * {@code embeddings=true}.
   *
   * @param i Batch output index (negative counts from end; {@code -1} = last token)
   * @return A new {@code float[nEmbdOut]} copy of the token embedding
   * @throws LlamaException if the native call returns a NULL pointer
   */
  public float[] getEmbeddingsIth(int i) {
    checkNotFreed();
    MemorySegment ptr = llama_get_embeddings_ith(segment, i);
    if (ptr == null || ptr.address() == 0) {
      throw new LlamaException(
        "llama_get_embeddings_ith returned NULL for index " +
          i +
          " – ensure the token was added with logits=true and embeddings=true was set on the context"
      );
    }
    int nEmbd = model.nEmbdOut();
    return ptr
      .reinterpret(nEmbd * ValueLayout.JAVA_FLOAT.byteSize())
      .toArray(ValueLayout.JAVA_FLOAT);
  }

  public MemorySegment getMemorySegment() {
    return segment;
  }

  public LlamaPerformance.ContextPerformance getPerformance(Arena arena) {
    checkNotFreed();
    MemorySegment perfData = llama_perf_context(arena, segment);
    return new LlamaPerformance.ContextPerformance(
      llama_perf_context_t_start_ms(perfData),
      llama_perf_context_t_load_ms(perfData),
      llama_perf_context_t_p_eval_ms(perfData),
      llama_perf_context_t_eval_ms(perfData),
      llama_perf_context_n_p_eval(perfData),
      llama_perf_context_n_eval(perfData),
      llama_perf_context_n_reused(perfData)
    );
  }

  /**
   * Collects log-probability information for the token at batch output index {@code batchIdx}.
   *
   * <p>Reads the raw logit vector produced by the last {@code decode()} call, applies
   * a numerically stable softmax to obtain probabilities, then returns the top-N entries
   * sorted by descending log-probability.  The sampled token ({@code sampledTokenId}) is
   * always included even if it did not make the top-N cut.
   *
   * <p>The caller must ensure the token at {@code batchIdx} was added to the batch with
   * {@code logits=true}, otherwise the returned segment will be NULL and this method
   * throws a {@link LlamaException}.
   *
   * @param vocab          The vocab used to decode token pieces
   * @param sampledTokenId The token that was sampled at this position
   * @param batchIdx       The index within the batch output (use {@code -1} for the last one)
   * @param topN           How many top alternatives to include (0 = chosen token only)
   * @return A {@link Logprobs} with the chosen token and up to {@code topN} top alternatives
   */
  public Logprobs getLogprobs(
    LlamaVocab vocab,
    int sampledTokenId,
    int batchIdx,
    int topN
  ) {
    checkNotFreed();
    int nVocab = nVocab();
    MemorySegment logitsPtr = llama_get_logits_ith(segment, batchIdx);
    if (logitsPtr == null || logitsPtr.address() == 0) {
      throw new LlamaException(
        "llama_get_logits_ith returned NULL – ensure the token was added with logits=true"
      );
    }

    // Read all nVocab logits into a float array.
    float[] logits = logitsPtr
      .reinterpret(nVocab * ValueLayout.JAVA_FLOAT.byteSize())
      .toArray(ValueLayout.JAVA_FLOAT);

    // Numerically-stable softmax: subtract max before exp to avoid overflow.
    float max = logits[0];
    for (float v : logits) {
      if (v > max) max = v;
    }
    double[] probs = new double[nVocab];
    double sum = 0.0;
    for (int i = 0; i < nVocab; i++) {
      probs[i] = Math.exp(logits[i] - max);
      sum += probs[i];
    }
    for (int i = 0; i < nVocab; i++) {
      probs[i] /= sum;
    }

    // Build top-N list using a simple partial sort (insertion into a small list).
    int limit = Math.min(topN, nVocab);
    // We always include the chosen token, so we need at least 1 slot.
    List<int[]> top = new ArrayList<>(limit + 1); // int[]{tokenId, rank} – rank unused; sorted inline
    // We'll store indices and sort by descending prob.
    // For efficiency we do a partial selection sort only up to `limit` elements.
    boolean[] visited = new boolean[nVocab];
    List<TokenLogprob> topList = new ArrayList<>(limit);
    for (int rank = 0; rank < limit; rank++) {
      int best = -1;
      for (int i = 0; i < nVocab; i++) {
        if (!visited[i] && (best == -1 || probs[i] > probs[best])) {
          best = i;
        }
      }
      visited[best] = true;
      topList.add(buildTokenLogprob(vocab, best, probs[best]));
    }

    // Ensure the chosen token is always in the list.
    boolean chosenPresent = topList
      .stream()
      .anyMatch(t -> t.tokenId() == sampledTokenId);
    if (!chosenPresent) {
      topList.add(
        buildTokenLogprob(vocab, sampledTokenId, probs[sampledTokenId])
      );
    }

    // The chosen token entry.
    TokenLogprob chosen = buildTokenLogprob(
      vocab,
      sampledTokenId,
      probs[sampledTokenId]
    );

    return new Logprobs(chosen, List.copyOf(topList));
  }

  private TokenLogprob buildTokenLogprob(
    LlamaVocab vocab,
    int tokenId,
    double probability
  ) {
    byte[] rawBytes = vocab.tokenToPiece(tokenId);
    String piece = rawBytes.length == 0
      ? ""
      : new String(rawBytes, StandardCharsets.UTF_8);
    double logprob = probability > 0.0
      ? Math.log(probability)
      : Double.NEGATIVE_INFINITY;

    List<Integer> byteList = new ArrayList<>(rawBytes.length);
    for (byte b : rawBytes) {
      byteList.add(b & 0xFF);
    }
    return new TokenLogprob(piece, tokenId, logprob, List.copyOf(byteList));
  }

  @Override
  public void free() {
    checkNotFreed();
    markFreed();
    llama_free(this);
  }
}
