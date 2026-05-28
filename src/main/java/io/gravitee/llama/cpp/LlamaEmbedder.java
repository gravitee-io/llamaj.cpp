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
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * High-level wrapper around {@link LlamaContext} for dense text embeddings.
 * <p>
 * Handles pooling, attention, tokenization, batch building and embedding extraction
 * so callers only deal with {@code String -> float[]}. Returns raw (un-normalised)
 * float vectors; any post-processing (L2 normalisation, quantization, etc.) is the
 * caller's responsibility.
 *
 * <h2>Auto-detection</h2>
 * When {@link Options#pooling()} or {@link Options#attention()} are {@code null} the
 * wrapper selects defaults based on the GGUF {@code general.architecture} metadata:
 * <ul>
 *   <li>Encoder architectures ({@code bert}, {@code nomic-bert}, {@code modern-bert},
 *       {@code jina-bert-*}, {@code neo-bert}, {@code eurobert}) default to
 *       {@link PoolingType#CLS} + {@link AttentionType#NON_CAUSAL}</li>
 *   <li>All other (decoder) architectures default to {@link PoolingType#LAST} +
 *       {@link AttentionType#CAUSAL}</li>
 * </ul>
 *
 * <h2>Thread safety</h2>
 * Not thread-safe. Create one instance per thread, or synchronise externally.
 *
 * <h2>Resource management</h2>
 * The caller always owns the {@link LlamaModel} and is responsible for freeing it.
 * Calling {@link #free()} or {@link #close()} on the embedder releases only the
 * internally-created {@link LlamaContext}; the passed-in model is untouched so it
 * can be reused across multiple embedder / reranker / classifier instances.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaEmbedder implements Freeable, AutoCloseable {

  private final LlamaModel model;
  private final LlamaContext context;
  private final LlamaVocab vocab;
  private final LlamaTokenizer tokenizer;
  private final PoolingType poolingType;

  private boolean freed = false;

  /**
   * Creates an embedding context on top of an externally-managed {@link LlamaModel}.
   * The caller retains ownership of the model and is responsible for freeing it.
   *
   * @param arena   Arena used for context params and vocab
   * @param model   Pre-loaded model (not freed on {@link #close()})
   * @param options Configuration; pass {@link Options#defaults()} for sensible defaults
   */
  public LlamaEmbedder(Arena arena, LlamaModel model, Options options) {
    this.model = model;

    String arch = model.metaVal(arena, "general.architecture");
    boolean encoder = TaskDefaults.isEncoderArch(arch);

    PoolingType pooling = options.pooling() != null
      ? options.pooling()
      : (encoder ? PoolingType.CLS : PoolingType.LAST);
    AttentionType attention = options.attention() != null
      ? options.attention()
      : (encoder ? AttentionType.NON_CAUSAL : AttentionType.CAUSAL);

    var ctxParams = new LlamaContextParams(arena)
      .embeddings(true)
      .poolingType(pooling)
      .attentionType(attention)
      .nCtx(options.nCtx() != null ? options.nCtx() : 0);
    if (options.nBatch() != null) ctxParams.nBatch(options.nBatch());
    if (options.nSeqMax() != null) ctxParams.nSeqMax(options.nSeqMax());

    this.context = new LlamaContext(arena, model, ctxParams);
    this.vocab = new LlamaVocab(model);
    this.tokenizer = new LlamaTokenizer(vocab, context);
    this.poolingType = pooling;
  }

  /**
   * Computes a dense embedding vector for the given text.
   *
   * @param text Input text; tokenised with the model's vocabulary
   * @return A fresh {@code float[nEmbdOut()]} copy of the pooled embedding
   * @throws LlamaException if decoding fails or embedding extraction returns null
   */
  public float[] embed(String text) {
    checkNotFreed();
    return embedAll(List.of(text)).getFirst();
  }

  /**
   * Computes embeddings for a list of texts, packing multiple sequences into a
   * single {@link LlamaBatch} whenever they fit under {@code nBatch} tokens and
   * {@code nSeqMax} sequences. Each packed batch is decoded once and the pooled
   * embedding for every sequence is collected via
   * {@link LlamaContext#getEmbeddingsSeq(int)}.
   * <p>
   * For long lists this is significantly faster than one {@code decode()} per
   * text because the transformer attention runs in parallel across sequences.
   *
   * @param texts Input texts
   * @return One embedding per input text, in the same order
   * @throws LlamaException if any single input exceeds {@code nBatch} tokens
   */
  public List<float[]> embedAll(List<String> texts) {
    checkNotFreed();
    if (texts.isEmpty()) {
      return List.of();
    }

    int n = texts.size();
    int nEmbd = model.nEmbdOut();
    int nBatch = context.nBatch();
    int nSeqMax = context.nSeqMax();
    float[][] results = new float[n][];

    try (Arena local = Arena.ofConfined()) {
      // Pre-tokenise everything so we can size batches by token count
      int[][] tokenIds = new int[n][];
      for (int i = 0; i < n; i++) {
        var tokenized = tokenizer.tokenize(local, texts.get(i));
        int size = tokenized.size();
        if (size > nBatch) {
          throw new LlamaException(
            "Input at index " +
              i +
              " has " +
              size +
              " tokens, exceeding nBatch=" +
              nBatch +
              ". Increase Options.nBatch or truncate the input."
          );
        }
        tokenIds[i] = tokenized
          .data()
          .reinterpret(size * ValueLayout.JAVA_INT.byteSize())
          .toArray(ValueLayout.JAVA_INT);
      }

      // Pack into batches and decode
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
            float[] emb = context.getEmbeddingsSeq(s);
            if (emb == null) {
              throw new LlamaException(
                "getEmbeddingsSeq returned null for pooling=" +
                  poolingType +
                  " - ensure the GGUF supports pooled embeddings for this pooling type"
              );
            }
            results[inputIndices[s]] = emb;
          }
        } finally {
          batch.free();
        }
      }
    }

    return List.of(results);
  }

  /** Output embedding dimension for this model. */
  public int nEmbdOut() {
    checkNotFreed();
    return model.nEmbdOut();
  }

  /** The pooling type active on this context (either explicit or auto-detected). */
  public PoolingType poolingType() {
    checkNotFreed();
    return poolingType;
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
      throw new LlamaException("LlamaEmbedder has been freed");
    }
  }

  /**
   * Configuration for {@link LlamaEmbedder}. Any {@code null} field triggers
   * auto-detection or a sensible default.
   *
   * @param nCtx      Context size in tokens; {@code null} -> {@code 0}, which makes llama.cpp
   *                  substitute the model's trained context length ({@code hparams.n_ctx_train})
   * @param nBatch    Batch size in tokens; {@code null} -> llama.cpp's default
   * @param nSeqMax   Maximum sequences packed per decode; {@code null} -> llama.cpp's default.
   *                  Higher values increase parallelism (at the cost of more
   *                  KV cache memory).
   * @param pooling   Pooling type; {@code null} -> auto (CLS for encoders, LAST for decoders)
   * @param attention Attention type; {@code null} -> auto (NON_CAUSAL for encoders, CAUSAL for decoders)
   */
  public record Options(
    Integer nCtx,
    Integer nBatch,
    Integer nSeqMax,
    PoolingType pooling,
    AttentionType attention
  ) {
    public static Options defaults() {
      return new Options(null, null, null, null, null);
    }

    public Options withNCtx(int nCtx) {
      return new Options(nCtx, nBatch, nSeqMax, pooling, attention);
    }

    public Options withNBatch(int nBatch) {
      return new Options(nCtx, nBatch, nSeqMax, pooling, attention);
    }

    public Options withNSeqMax(int nSeqMax) {
      return new Options(nCtx, nBatch, nSeqMax, pooling, attention);
    }

    public Options withPooling(PoolingType pooling) {
      return new Options(nCtx, nBatch, nSeqMax, pooling, attention);
    }

    public Options withAttention(AttentionType attention) {
      return new Options(nCtx, nBatch, nSeqMax, pooling, attention);
    }
  }
}
