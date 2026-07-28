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

  /**
   * Whether the owning context shares one KV buffer across sequences. Recorded here because
   * {@link #copyPrefix} is only legal when it is {@code true}, and violating that aborts the
   * process rather than throwing.
   */
  private final boolean kvUnified;

  public LlamaMemory(LlamaContext context) {
    super(llama_get_memory(context.segment));
    this.kvUnified = context.isKvUnified();
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
   * Republishes the leading {@code matchedTokens} KV rows of {@code seqIdSrc} onto
   * {@code seqIdDst}, and returns the prefix length the destination may then reuse — the value to
   * pass to {@link ConversationState#initialize(String, int)}.
   *
   * <p>This is the cross-request prefix cache in one call, and it <b>requires a unified KV
   * cache</b> ({@link LlamaContextParams#kvUnified(boolean)}). Under a unified cache there is a
   * single stream, so every sequence indexes the same cell pool and a cell carries a <em>set</em>
   * of sequence ids: the copy adds {@code seqIdDst} to the cells at {@code [0, matchedTokens)} and
   * moves no tensor data. Publishing a 2000-token system prompt to a second conversation then
   * costs microseconds and no extra VRAM, and the source may still be generating — cells it shares
   * are not disturbed, and it never rewinds into its own prompt.
   *
   * <p>Without a unified cache each sequence owns its own stream and llama.cpp must physically
   * copy buffer data, which it supports only for a whole sequence: a partial range trips
   * {@code GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")}, and a failed
   * GGML assert calls {@code abort()} — it takes the JVM down with SIGABRT, uncatchably. This
   * method therefore refuses up front with a {@link LlamaException} rather than letting the process
   * die. Copying the whole sequence and trimming afterwards is not a workaround: on that path the
   * copy is real, so it would cost full KV memory and full copy time, defeating the purpose.
   *
   * <p>The two rules this method exists to enforce:
   * <ul>
   *   <li>The destination is wiped first. {@code seq_cp} adds rows rather than replacing them, so
   *       copying onto a sequence that still holds cells would stack one prefix on another.</li>
   *   <li>The result is clamped to {@code promptTokens - 1}. The final prompt token must always be
   *       re-decoded to produce the logits its first output token is sampled from, so claiming it
   *       as reused is never useful. {@code initialize} clamps too, but clamping here keeps the
   *       rows we copy and the prefill we ask for in agreement.</li>
   * </ul>
   *
   * <p><b>Caller invariant.</b> Whatever tracks which tokens a sequence holds must never advertise
   * more than is provably resident, or a later copy reads cells that no longer exist. A sequence
   * being prefilled holds only what was copied onto it; widen the record to
   * {@link ConversationState#committedTokens()} — exactly positions {@code [0, nPast)} — only once
   * the prefill has run. Retaining the source's KV is the caller's job too: mark its state
   * {@link ConversationState#setRetainKv(boolean)} or remove it with {@code keepKv}, otherwise the
   * rows are wiped when it finishes and the advertised prefix becomes a dangling claim.
   *
   * <p>Prefix reuse can still be refused at prefill time by recurrent and hybrid models, which
   * cannot always rewind far enough for the partial trim; that path falls back to a cold full
   * prefill and reports {@link ConversationState#isPrefixReuseHonored()} {@code == false}. Correct,
   * just slower.
   *
   * @param seqIdSrc      Sequence holding the prefix
   * @param seqIdDst      Sequence to publish it onto; its existing rows are discarded
   * @param matchedTokens Leading tokens the two prompts share
   * @param promptTokens  Length of the destination's tokenized prompt
   * @return Prefix tokens the destination may reuse, {@code 0} when nothing was worth copying
   *
   * <p>Example — serve a prompt that shares a system prompt with a busy conversation:
   * <pre>{@code
   * int shared = commonPrefixLength(donorTokens, promptTokens);
   * int reuse = memory.copyPrefix(donorSeqId, freeSeqId, shared, promptTokens.length);
   * ConversationState.create(arena, ctx, tokenizer, sampler, freeSeqId)
   *   .setRetainKv(true)
   *   .initialize(prompt, reuse);   // prefills only the suffix
   * }</pre>
   */
  public int copyPrefix(
    int seqIdSrc,
    int seqIdDst,
    int matchedTokens,
    int promptTokens
  ) {
    if (seqIdSrc == seqIdDst) {
      throw new LlamaException(
        "copyPrefix source and destination must differ (got " + seqIdSrc + ")"
      );
    }
    if (!kvUnified) {
      // Refuse rather than let GGML_ASSERT abort() the JVM on the cross-stream path.
      throw new LlamaException(
        "copyPrefix requires a unified KV cache: build the context with " +
          "LlamaContextParams.kvUnified(true). Without it each sequence owns its own KV stream, " +
          "llama_memory_seq_cp has to copy buffer data and only supports whole sequences, and a " +
          "partial range aborts the process instead of failing."
      );
    }
    int reuse = Math.min(matchedTokens, promptTokens - 1);
    seqRm(seqIdDst, -1, -1);
    if (reuse <= 0) {
      return 0;
    }
    seqCp(seqIdSrc, seqIdDst, 0, reuse);
    return reuse;
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
