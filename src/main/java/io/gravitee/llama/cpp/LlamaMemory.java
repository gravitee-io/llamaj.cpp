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

/**
 * Manages the KV cache memory for one or more sequences.
 * The KV cache stores the key-value pairs from previous tokens to enable efficient generation.
 *
 * <p>For single-sequence usage (default sequence ID 0):
 * <pre>{@code
 * memory.clear();  // Clear all cached tokens
 * int min = memory.posMin();  // Get minimum position in cache
 * int max = memory.posMax();  // Get maximum position in cache
 * }</pre>
 *
 * <p>For multi-sequence usage:
 * <pre>{@code
 * // Clear specific sequence
 * memory.seqRm(1, -1, -1);
 *
 * // Copy sequence 0 to sequences 1, 2, 3 (e.g., shared system prompt)
 * memory.seqCp(0, 1, -1, -1);
 * memory.seqCp(0, 2, -1, -1);
 * memory.seqCp(0, 3, -1, -1);
 *
 * // Keep only sequence 0, remove all others
 * memory.seqKeep(0);
 * }</pre>
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class LlamaMemory extends MemorySegmentAware {

  public LlamaMemory(LlamaContext context) {
    super(llama_get_memory(context.segment));
  }

  /**
   * Gets the minimum position in the cache for the default sequence (ID 0).
   */
  public int posMin() {
    return posMin(0);
  }

  /**
   * Gets the minimum position in the cache for a specific sequence.
   * @param seqId The sequence ID
   * @return The minimum position, or -1 if the sequence is empty
   */
  public int posMin(int seqId) {
    return llama_memory_seq_pos_min(this.segment, seqId);
  }

  /**
   * Gets the maximum position in the cache for the default sequence (ID 0).
   */
  public int posMax() {
    return posMax(0);
  }

  /**
   * Gets the maximum position in the cache for a specific sequence.
   * @param seqId The sequence ID
   * @return The maximum position, or -1 if the sequence is empty
   */
  public int posMax(int seqId) {
    return llama_memory_seq_pos_max(this.segment, seqId);
  }

  /**
   * Clears all memory (all sequences).
   */
  public void clear() {
    llama_memory_clear(this.segment, true);
  }

  /**
   * Removes all tokens that belong to the specified sequence and have positions in [p0, p1).
   * Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails.
   *
   * @param seqId The sequence ID (< 0 to match any sequence)
   * @param p0 Start position (< 0 means [0, p1])
   * @param p1 End position (< 0 means [p0, inf))
   * @return true on success, false if a partial sequence cannot be removed
   *
   * <p>Examples:
   * <pre>{@code
   * // Remove entire sequence when conversation finishes (most common use case)
   * if (vocab.isEog(token) || reachedMaxTokens) {
   *   memory.seqRm(conversationId, -1, -1);  // Free KV cache for this conversation
   * }
   *
   * // Remove specific position range
   * memory.seqRm(0, 10, 20);  // Remove tokens at positions [10, 20) from sequence 0
   *
   * // Clear all sequences
   * memory.seqRm(-1, -1, -1); // Remove all tokens from all sequences
   * }</pre>
   */
  public boolean seqRm(int seqId, int p0, int p1) {
    return llama_memory_seq_rm(this.segment, seqId, p0, p1);
  }

  /**
   * Copies all tokens that belong to the specified sequence to another sequence.
   * This is useful for sharing common prefixes (like system prompts) across multiple sequences.
   *
   * @param seqIdSrc Source sequence ID
   * @param seqIdDst Destination sequence ID
   * @param p0 Start position (< 0 means [0, p1])
   * @param p1 End position (< 0 means [p0, inf))
   *
   * <p>Example - Share system prompt across 4 sequences:
   * <pre>{@code
   * // Process system prompt in sequence 0
   * batch.add(systemToken, 0, List.of(0), false);
   * context.decode(batch);
   *
   * // Copy to sequences 1, 2, 3
   * memory.seqCp(0, 1, -1, -1);
   * memory.seqCp(0, 2, -1, -1);
   * memory.seqCp(0, 3, -1, -1);
   * }</pre>
   */
  public void seqCp(int seqIdSrc, int seqIdDst, int p0, int p1) {
    llama_memory_seq_cp(this.segment, seqIdSrc, seqIdDst, p0, p1);
  }

  /**
   * Removes all tokens that do not belong to the specified sequence.
   * @param seqId The sequence ID to keep
   */
  public void seqKeep(int seqId) {
    llama_memory_seq_keep(this.segment, seqId);
  }

  /**
   * Adds relative position delta to all tokens that belong to the specified sequence
   * and have positions in [p0, p1).
   *
   * @param seqId The sequence ID
   * @param p0 Start position (< 0 means [0, p1])
   * @param p1 End position (< 0 means [p0, inf))
   * @param delta The position delta to add
   */
  public void seqAdd(int seqId, int p0, int p1, int delta) {
    llama_memory_seq_add(this.segment, seqId, p0, p1, delta);
  }

  /**
   * Performs integer division of the positions by factor d > 1.
   *
   * @param seqId The sequence ID
   * @param p0 Start position (< 0 means [0, p1])
   * @param p1 End position (< 0 means [p0, inf))
   * @param d The divisor (must be > 1)
   */
  public void seqDiv(int seqId, int p0, int p1, int d) {
    llama_memory_seq_div(this.segment, seqId, p0, p1, d);
  }
}
