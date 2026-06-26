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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_sampler_apply;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;

/**
 * Reusable candidate buffer mirroring the native {@code llama_token_data_array}.
 *
 * <p>Holds an {@code n_vocab}-sized array of {@code llama_token_data} records plus the
 * array header. Unlike {@link LlamaSampler#sample}, which reads the context logits and
 * returns only a token id, this lets a caller feed an arbitrary logit row through the
 * sampler chain via {@link LlamaRuntime#llama_sampler_apply} and read back both the
 * selected token and its probability. {@link DiffusionGenerator} uses it once per masked
 * position per denoising step, so the buffer is allocated once and refilled in place.
 *
 * <p>Layouts are written with raw {@link ValueLayout} accessors that match the
 * jextract-generated {@code llama_token_data} ({@code {int id; float logit; float p;}},
 * 12 bytes) and {@code llama_token_data_array}
 * ({@code {token_data* data; size_t size; int64_t selected; bool sorted;}}, 32 bytes)
 * struct layouts on all supported 64-bit platforms.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaTokenDataArray {

  // llama_token_data: { llama_token id; float logit; float p; }
  private static final long TD_ID_OFFSET = 0;
  private static final long TD_LOGIT_OFFSET = 4;
  private static final long TD_P_OFFSET = 8;
  private static final long TD_SIZE = 12;

  // llama_token_data_array: { llama_token_data* data; size_t size; int64_t selected; bool sorted; }
  private static final long ARR_DATA_OFFSET = 0;
  private static final long ARR_SIZE_OFFSET = 8;
  private static final long ARR_SELECTED_OFFSET = 16;
  private static final long ARR_SORTED_OFFSET = 24;
  private static final long ARR_SIZE = 32;

  private final int nVocab;
  private final MemorySegment data; // n_vocab * llama_token_data
  private final MemorySegment array; // the llama_token_data_array header

  public LlamaTokenDataArray(SegmentAllocator allocator, int nVocab) {
    this.nVocab = nVocab;
    this.data = allocator.allocate(TD_SIZE * nVocab, 4);
    this.array = allocator.allocate(ARR_SIZE, 8);
    array.set(ValueLayout.ADDRESS, ARR_DATA_OFFSET, data);
    array.set(ValueLayout.JAVA_LONG, ARR_SIZE_OFFSET, nVocab);
  }

  /**
   * Populates the candidate buffer from a logit row and resets the array header so the
   * sampler chain treats every token as a fresh, unsorted candidate.
   *
   * @param logits     A segment over at least {@code n_vocab} floats (one decoded row)
   * @param rowOffset  Float offset into {@code logits} where the row begins
   */
  public void fill(MemorySegment logits, long rowOffset) {
    for (int id = 0; id < nVocab; id++) {
      long base = (long) id * TD_SIZE;
      float logit = logits.getAtIndex(ValueLayout.JAVA_FLOAT, rowOffset + id);
      data.set(ValueLayout.JAVA_INT, base + TD_ID_OFFSET, id);
      data.set(ValueLayout.JAVA_FLOAT, base + TD_LOGIT_OFFSET, logit);
      data.set(ValueLayout.JAVA_FLOAT, base + TD_P_OFFSET, 0.0f);
    }
    array.set(ValueLayout.JAVA_LONG, ARR_SIZE_OFFSET, nVocab);
    array.set(ValueLayout.JAVA_LONG, ARR_SELECTED_OFFSET, -1L);
    array.set(ValueLayout.JAVA_BYTE, ARR_SORTED_OFFSET, (byte) 0);
  }

  /** Runs the sampler chain in place over the current candidates. */
  public void apply(LlamaSampler sampler) {
    llama_sampler_apply(sampler.segment, array);
  }

  /** Index selected by the last {@link #apply}, or {@code -1} if none. */
  public long selectedIndex() {
    return array.get(ValueLayout.JAVA_LONG, ARR_SELECTED_OFFSET);
  }

  /**
   * Current number of live candidates. After {@link #apply}, truncating samplers
   * (top-k / top-p) may shrink this below {@code nVocab}; entries beyond it are stale.
   */
  public long size() {
    return array.get(ValueLayout.JAVA_LONG, ARR_SIZE_OFFSET);
  }

  private long selectedByteBase() {
    long sel = selectedIndex();
    if (sel < 0) {
      throw new LlamaException("No token selected; call apply() first");
    }
    return sel * TD_SIZE;
  }

  /** Token id of the selected candidate. */
  public int selectedId() {
    return data.get(ValueLayout.JAVA_INT, selectedByteBase() + TD_ID_OFFSET);
  }

  /** Probability of the selected candidate (its confidence). */
  public float selectedProbability() {
    return data.get(ValueLayout.JAVA_FLOAT, selectedByteBase() + TD_P_OFFSET);
  }

  /** Probability of the candidate at {@code index} after {@link #apply}. */
  public float probabilityAt(long index) {
    return data.get(ValueLayout.JAVA_FLOAT, index * TD_SIZE + TD_P_OFFSET);
  }

  /**
   * Token id of the candidate at {@code index}. After {@link #apply} the array is sorted /
   * truncated by the sampler chain, so the index no longer equals the token id.
   */
  public int idAt(long index) {
    return data.get(ValueLayout.JAVA_INT, index * TD_SIZE + TD_ID_OFFSET);
  }

  public int nVocab() {
    return nVocab;
  }
}
