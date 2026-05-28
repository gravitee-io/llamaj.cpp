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

import static io.gravitee.llama.cpp.AttentionType.CAUSAL;
import static io.gravitee.llama.cpp.AttentionType.NON_CAUSAL;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * High-level wrapper around {@link LlamaContext} for cross-encoder reranking.
 * <p>
 * A reranker produces a relevance score for a (query, document) pair. The output
 * dimension depends on the underlying model:
 * <ul>
 *   <li>BERT-family cross-encoders (BGE reranker, Jina reranker) return a single
 *       float per pair: a raw relevance logit. Apply sigmoid to map to [0,1].</li>
 *   <li>Qwen3-Reranker returns two floats: P(yes) and P(no) softmaxed. The
 *       relevance probability is {@code scores[0]}.</li>
 * </ul>
 * This wrapper returns the raw {@code float[]} - the caller decides how to
 * interpret it via {@link #nClsOut()}.
 *
 * <h2>Input formatting</h2>
 * Different reranker families require different input formats (plain concatenation,
 * chat templates, etc.). The wrapper delegates this to a {@link RerankTemplate}
 * supplied via {@link Options#template()}. When {@code null} the wrapper uses
 * {@link RerankTemplate#PLAIN} which is correct for BERT-family cross-encoders.
 *
 * <h2>Attention auto-detection</h2>
 * When {@link Options#attention()} is {@code null} the wrapper selects
 * {@link AttentionType#NON_CAUSAL} for encoder architectures and
 * {@link AttentionType#CAUSAL} for decoder architectures.
 *
 * <h2>Resource management</h2>
 * The caller always owns the {@link LlamaModel} and is responsible for freeing it.
 * {@link #free()} / {@link #close()} releases only the internally-created
 * {@link LlamaContext}.
 *
 * <h2>Thread safety</h2>
 * Not thread-safe. Create one instance per thread, or synchronise externally.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaReranker implements Freeable, AutoCloseable {

  private final LlamaModel model;
  private final LlamaContext context;
  private final LlamaVocab vocab;
  private final LlamaTokenizer tokenizer;
  private final RerankTemplate template;
  private final int nClsOut;

  private boolean freed = false;

  /**
   * Creates a reranker context on top of an externally-managed {@link LlamaModel}.
   * The caller retains ownership of the model and is responsible for freeing it.
   *
   * @param arena   Arena used for context params and vocab
   * @param model   Pre-loaded model (not freed on {@link #close()})
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   */
  public LlamaReranker(Arena arena, LlamaModel model, Options options) {
    this.model = model;
    this.nClsOut = model.nClsOut();

    String arch = model.metaVal(arena, "general.architecture");
    boolean encoder = TaskDefaults.isEncoderArch(arch);
    AttentionType attention = options.attention() != null
      ? options.attention()
      : (encoder ? NON_CAUSAL : CAUSAL);

    this.template = options.template() != null
      ? options.template()
      : RerankTemplate.PLAIN;

    var ctxParams = new LlamaContextParams(arena)
      .embeddings(true)
      .poolingType(PoolingType.RANK)
      .attentionType(attention)
      .nCtx(options.nCtx() != null ? options.nCtx() : 0);
    if (options.nBatch() != null) ctxParams.nBatch(options.nBatch());
    if (options.nSeqMax() != null) ctxParams.nSeqMax(options.nSeqMax());

    this.context = new LlamaContext(arena, model, ctxParams);
    this.vocab = new LlamaVocab(model);
    this.tokenizer = new LlamaTokenizer(vocab, context);
  }

  /**
   * Returns the raw relevance score(s) for a (query, document) pair.
   * The returned array has size {@link #nClsOut()} - typically 1 for BERT
   * cross-encoders or 2 for Qwen3-Reranker.
   *
   * @param query    The search query
   * @param document The candidate document
   * @return A fresh {@code float[]} with the raw reranker output
   */
  public float[] score(String query, String document) {
    checkNotFreed();
    return scoreAll(query, List.of(document)).get(0);
  }

  /**
   * Scores many documents against a single query, packing multiple (query, document)
   * pairs into a single {@link LlamaBatch} whenever they fit under {@code nBatch}
   * tokens and {@code nSeqMax} sequences. Each packed batch is decoded once and
   * the per-sequence scores are collected via
   * {@link LlamaContext#getEmbeddingsSeq(int)}.
   * <p>
   * For large document lists this is significantly faster than one
   * {@code decode()} per document because the transformer attention runs in
   * parallel across sequences.
   *
   * @param query     The search query
   * @param documents Candidate documents
   * @return One raw score array per document, in input order
   * @throws LlamaException if any single formatted input exceeds {@code nBatch} tokens
   */
  public List<float[]> scoreAll(String query, List<String> documents) {
    checkNotFreed();
    if (documents.isEmpty()) {
      return List.of();
    }

    int n = documents.size();
    int nBatch = context.nBatch();
    int nSeqMax = context.nSeqMax();
    float[][] results = new float[n][];

    try (Arena local = Arena.ofConfined()) {
      // Pre-tokenise every (query, document) pair once
      int[][] tokenIds = new int[n][];
      for (int i = 0; i < n; i++) {
        String input = template.format(query, documents.get(i));
        var tokenized = tokenizer.tokenize(local, input);
        int size = tokenized.size();
        if (size > nBatch) {
          throw new LlamaException(
            "Formatted (query, document) pair at index " +
              i +
              " tokenises to " +
              size +
              " tokens, exceeding nBatch=" +
              nBatch +
              ". Increase Options.nBatch or truncate the document."
          );
        }
        tokenIds[i] = tokenized
          .data()
          .reinterpret(size * ValueLayout.JAVA_INT.byteSize())
          .toArray(ValueLayout.JAVA_INT);
      }

      int i = 0;
      while (i < n) {
        context.clearCache();
        var batch = new LlamaBatch(local, nBatch, 0, nSeqMax);
        try {
          batch.enableCache();

          int tokensInBatch = 0;
          int seqsInBatch = 0;
          int[] inputIndices = new int[nSeqMax];

          while (i < n && seqsInBatch < nSeqMax) {
            int[] ids = tokenIds[i];
            if (tokensInBatch + ids.length > nBatch) {
              break;
            }
            List<Integer> seqIdList = List.of(seqsInBatch);
            for (int k = 0; k < ids.length; k++) {
              batch.add(ids[k], k, seqIdList, true);
            }
            inputIndices[seqsInBatch] = i;
            tokensInBatch += ids.length;
            seqsInBatch++;
            i++;
          }

          int ret = context.decode(batch);
          if (ret != 0) {
            throw new LlamaException(
              "decode() returned non-zero status: " + ret
            );
          }

          for (int s = 0; s < seqsInBatch; s++) {
            float[] scores = context.getEmbeddingsSeq(s);
            if (scores == null) {
              throw new LlamaException(
                "getEmbeddingsSeq returned null - ensure the GGUF supports RANK " +
                  "pooling (classifier head must be present in the file)"
              );
            }
            results[inputIndices[s]] = scores;
          }
        } finally {
          batch.free();
        }
      }
    }

    return List.of(results);
  }

  /** Number of classifier outputs - 1 for BERT rerankers, 2 for Qwen3-Reranker. */
  public int nClsOut() {
    checkNotFreed();
    return nClsOut;
  }

  /** The input formatting template in use (either explicit or {@link RerankTemplate#PLAIN}). */
  public RerankTemplate template() {
    checkNotFreed();
    return template;
  }

  /** Exposes the underlying context for advanced use cases. */
  public LlamaContext context() {
    checkNotFreed();
    return context;
  }

  /** Exposes the underlying model. The caller is responsible for freeing it. */
  public LlamaModel model() {
    checkNotFreed();
    return model;
  }

  @Override
  public void free() {
    if (freed) return;
    freed = true;
    context.free();
  }

  @Override
  public boolean isFree() {
    return freed;
  }

  @Override
  public void close() {
    free();
  }

  private void checkNotFreed() {
    if (freed) {
      throw new LlamaException("LlamaReranker has been freed");
    }
  }

  /**
   * Configuration for {@link LlamaReranker}. Any {@code null} field triggers
   * auto-detection or a sensible default.
   *
   * @param nCtx      Context size in tokens; {@code null} -> {@code 0}, which makes llama.cpp
   *                  substitute the model's trained context length ({@code hparams.n_ctx_train})
   * @param nBatch    Batch size in tokens; {@code null} -> llama.cpp's default
   * @param nSeqMax   Maximum sequences packed per decode; {@code null} -> llama.cpp's default.
   *                  Higher values increase parallelism (at the cost of more
   *                  KV cache memory).
   * @param attention Attention type; {@code null} -> auto (NON_CAUSAL for encoders, CAUSAL for decoders)
   * @param template  Input formatting template; {@code null} -> {@link RerankTemplate#PLAIN}
   */
  public record Options(
    Integer nCtx,
    Integer nBatch,
    Integer nSeqMax,
    AttentionType attention,
    RerankTemplate template
  ) {
    public static Options defaults() {
      return new Options(null, null, null, null, null);
    }

    public Options withNCtx(int nCtx) {
      return new Options(nCtx, nBatch, nSeqMax, attention, template);
    }

    public Options withNBatch(int nBatch) {
      return new Options(nCtx, nBatch, nSeqMax, attention, template);
    }

    public Options withNSeqMax(int nSeqMax) {
      return new Options(nCtx, nBatch, nSeqMax, attention, template);
    }

    public Options withAttention(AttentionType attention) {
      return new Options(nCtx, nBatch, nSeqMax, attention, template);
    }

    public Options withTemplate(RerankTemplate template) {
      return new Options(nCtx, nBatch, nSeqMax, attention, template);
    }
  }
}
