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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_model_load_from_file;

import java.lang.foreign.*;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaModel extends MemorySegmentAware implements Freeable {

  private LlamaLoraAdapter loraAdapter;

  public LlamaModel(
    SegmentAllocator arena,
    Path modelPath,
    LlamaModelParams params
  ) {
    this(
      llama_model_load_from_file(
        getModelAsString(arena, modelPath),
        params.segment
      )
    );
    if (segment == null || segment.address() == 0) {
      throw new LlamaException("Failed to load model: " + modelPath);
    }
  }

  public LlamaModel(MemorySegment segment) {
    super(segment);
  }

  private static MemorySegment getModelAsString(
    SegmentAllocator arena,
    Path modelPath
  ) {
    return arena.allocateFrom(modelPath.toAbsolutePath().toString());
  }

  public LlamaModel initLoraAdapter(Arena arena, Path loraPath) {
    this.loraAdapter = new LlamaLoraAdapter(arena, this, loraPath);
    return this;
  }

  /**
   * Returns the output embedding dimension of the model.
   * Use this to size {@code float[]} buffers when reading embedding vectors via
   * {@link LlamaContext#getEmbeddingsIth} or {@link LlamaContext#getEmbeddingsSeq}.
   * For most models this equals the hidden size, but for classifier / reranker models
   * it may differ.
   *
   * @return The output embedding dimension
   */
  public int nEmbdOut() {
    checkNotFreed();
    return LlamaRuntime.llama_model_n_embd_out(segment);
  }

  /**
   * Returns the model's trained context length, as reported by llama.cpp
   * (mirrors the GGUF {@code {arch}.context_length} metadata).
   */
  public int nCtxTrain() {
    checkNotFreed();
    return LlamaRuntime.llama_model_n_ctx_train(segment);
  }

  /**
   * Returns the number of classifier output classes.
   * When {@link PoolingType#RANK} is active and this returns {@code 1}, the model is
   * a reranker (single relevance score per sequence). When {@code > 1}, the model
   * exposes multiple labelled output classes.
   * Returns {@code 0} for generative / embedding-only models.
   *
   * @return The number of classifier outputs
   */
  public int nClsOut() {
    checkNotFreed();
    return LlamaRuntime.llama_model_n_cls_out(segment);
  }

  /**
   * Returns the label string for classifier output index {@code i}, or {@code null}
   * if the model does not embed class labels.
   *
   * @param arena Used to allocate the temporary C string buffer
   * @param i     The classifier output index (0-based, must be {@code < nClsOut()})
   * @return The label string, or {@code null} if not available
   */
  public String clsLabel(Arena arena, int i) {
    checkNotFreed();
    return LlamaRuntime.llama_model_cls_label(arena, segment, i);
  }

  /**
   * Reads a GGUF metadata value by key (e.g. {@code "general.architecture"}).
   * Returns {@code null} if the key is not present in the model metadata.
   *
   * @param arena Used to allocate the C string buffer
   * @param key   The GGUF metadata key
   * @return The value string, or {@code null} if not found
   */
  public String metaVal(Arena arena, String key) {
    checkNotFreed();
    return LlamaRuntime.llama_model_meta_val_str(arena, segment, key);
  }

  /**
   * Returns the total number of GGUF metadata key/value pairs in the model.
   *
   * @return The number of metadata entries
   */
  public int metaCount() {
    checkNotFreed();
    return LlamaRuntime.llama_model_meta_count(segment);
  }

  /**
   * Returns all GGUF metadata as an ordered map of key → value strings.
   * Insertion order matches the GGUF file order.
   *
   * @param arena Used to allocate C string buffers for each key and value
   * @return A {@link LinkedHashMap} of all metadata entries
   */
  public Map<String, String> meta(Arena arena) {
    checkNotFreed();
    int count = LlamaRuntime.llama_model_meta_count(segment);
    Map<String, String> result = new LinkedHashMap<>(count);
    for (int i = 0; i < count; i++) {
      String key = LlamaRuntime.llama_model_meta_key_by_index(
        arena,
        segment,
        i
      );
      String val = LlamaRuntime.llama_model_meta_val_str_by_index(
        arena,
        segment,
        i
      );
      if (key != null) {
        result.put(key, val);
      }
    }
    return result;
  }

  /**
   * Returns a human-readable description of the model (architecture, quantization, size, etc.)
   * as reported by llama.cpp from the GGUF metadata.
   *
   * @param arena Used to allocate the C string buffer
   * @return The description string
   */
  public String desc(Arena arena) {
    checkNotFreed();
    return LlamaRuntime.llama_model_desc(arena, segment);
  }

  /**
   * Returns {@code true} if this is a diffusion-based model (e.g. LLaDA, Dream)
   * that generates text by iterative denoising rather than autoregressive
   * decoding. Use {@link DiffusionGenerator} to run such models.
   *
   * @return {@code true} for diffusion models, {@code false} otherwise
   */
  public boolean isDiffusion() {
    checkNotFreed();
    return LlamaRuntime.llama_model_is_diffusion(segment);
  }

  @Override
  public void free() {
    checkNotFreed();
    if (loraAdapter != null) {
      loraAdapter.free();
    }
    markFreed();
    LlamaRuntime.llama_model_free(this);
  }
}
