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
import static java.lang.foreign.ValueLayout.JAVA_INT;

import io.gravitee.llama.cpp.LlamaTokenizer.TokenizerResponse;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.util.List;

/**
 * Represents a batch of tokens that can be processed together.
 * Supports both single-sequence and multi-sequence batching for parallel processing.
 * Includes optional field caching for optimized performance when adding many tokens.
 *
 * <p>Key concepts:
 * <ul>
 *   <li><b>Batch</b>: A collection of tokens to be processed in a single forward pass</li>
 *   <li><b>Sequence ID</b>: Identifies which conversation/context a token belongs to</li>
 *   <li><b>Position</b>: The position of the token within its sequence</li>
 *   <li><b>Field Cache</b>: Optional cache for eliminating reflection overhead when adding many tokens</li>
 * </ul>
 *
 * <p>For multi-sequence processing:
 * <pre>{@code
 * // Create a batch that can hold multiple tokens and sequences
 * var batch = new LlamaBatch(arena, 128, 0, 4);
 *
 * // Add tokens from different sequences
 * batch.add(token1, pos1, List.of(0), true);  // sequence 0
 * batch.add(token2, pos2, List.of(1), true);  // sequence 1
 * batch.add(token3, pos3, List.of(0, 1), true); // shared between sequences 0 and 1
 * }</pre>
 *
 * <p>For optimized performance with many tokens:
 * <pre>{@code
 * // Enable field caching to eliminate reflection overhead
 * var batch = new LlamaBatch(arena, 512, 0, 1);
 * batch.enableCache();  // Create cache once
 *
 * // Add many tokens efficiently (no reflection overhead)
 * for (int i = 0; i < 512; i++) {
 *   batch.add(tokenIds[i], i, List.of(0), i == 511);
 * }
 * }</pre>
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaBatch extends MemorySegmentAware implements Freeable {

  private BatchFieldCache fieldCache;

  /**
   * Creates a batch from a tokenizer response (single sequence with ID 0).
   */
  public LlamaBatch(
    SegmentAllocator allocator,
    TokenizerResponse tokenizerResponse
  ) {
    this(allocator, tokenizerResponse.data(), tokenizerResponse.size());
  }

  /**
   * Creates a batch from a tokenizer response with a specific sequence ID.
   * Use this for multi-sequence scenarios where you need to process prompts with non-zero sequence IDs.
   *
   * @param allocator The memory allocator
   * @param tokenizerResponse The tokenized prompt
   * @param sequenceId The sequence ID for this batch
   */
  public LlamaBatch(
    SegmentAllocator allocator,
    TokenizerResponse tokenizerResponse,
    int sequenceId
  ) {
    super(llama_batch_init(allocator, tokenizerResponse.size(), 0, 1));
    // Add all tokens from the tokenizer response with the specified sequence ID
    for (int i = 0; i < tokenizerResponse.size(); i++) {
      int tokenId = tokenizerResponse.data().getAtIndex(JAVA_INT, i);
      add(tokenId, i, List.of(sequenceId), i == tokenizerResponse.size() - 1);
    }
  }

  /**
   * Creates a batch with a single token (single sequence with ID 0).
   */
  public LlamaBatch(SegmentAllocator allocator, int tokenId) {
    this(allocator, getTokenArray(allocator, tokenId), 1);
  }

  /**
   * Creates a batch from a token array (single sequence with ID 0).
   * Uses the legacy llama_batch_get_one helper for simple single-sequence batches.
   */
  public LlamaBatch(
    SegmentAllocator allocator,
    MemorySegment segment,
    int size
  ) {
    super(llama_batch_get_one(allocator, segment, size));
  }

  /**
   * Creates an advanced batch that can hold multiple tokens and sequences.
   * This constructor provides full control for parallel processing of multiple conversations.
   *
   * @param allocator The memory allocator
   * @param nTokens The maximum number of tokens this batch can hold
   * @param embd Embedding size (0 for token-based input, otherwise allocates space for embeddings)
   * @param nSeqMax The maximum number of sequence IDs that can be assigned to a single token
   *
   * <p>Example: Create a batch for processing 4 parallel conversations with up to 128 tokens:
   * <pre>{@code
   * var batch = new LlamaBatch(arena, 128, 0, 4);
   * }</pre>
   */
  public LlamaBatch(
    SegmentAllocator allocator,
    int nTokens,
    int embd,
    int nSeqMax
  ) {
    super(llama_batch_init(allocator, nTokens, embd, nSeqMax));
  }

  private static MemorySegment getTokenArray(
    SegmentAllocator allocator,
    int tokenId
  ) {
    var tokenArray = allocator.allocate(JAVA_INT, 1);
    tokenArray.set(JAVA_INT, 0, tokenId);
    return tokenArray;
  }

  /**
   * Enables field caching for this batch.
   * Call this before adding many tokens to eliminate reflection overhead.
   *
   * <p><b>Performance Impact:</b>
   * <ul>
   *   <li>First call: 5 reflection calls to cache batch fields
   *   <li>Subsequent calls: 0 reflection calls (uses cached references)
   *   <li>Speedup: ~600× faster for large batches (512+ tokens)
   * </ul>
   *
   * <p>Example:
   * <pre>{@code
   * var batch = new LlamaBatch(arena, 512, 0, 1);
   * batch.enableCache();  // 5 reflections happen here
   * for (int i = 0; i < 512; i++) {
   *   batch.add(tokenIds[i], i, seqIds, logits);  // 0 reflections per add!
   * }
   * }</pre>
   */
  public void enableCache() {
    if (this.fieldCache == null) {
      this.fieldCache = new BatchFieldCache(this.segment);
    }
  }

  /**
   * Gets the field cache for this batch, creating it if necessary.
   * Used internally by add() to enable optimized token addition.
   */
  private BatchFieldCache getOrCreateCache() {
    if (this.fieldCache == null) {
      this.fieldCache = new BatchFieldCache(this.segment);
    }
    return this.fieldCache;
  }

  /**
   * Adds a token to this batch for processing.
   *
   * @param token The token ID to add
   * @param pos The position of this token in the sequence (typically n_past)
   * @param seqIds The list of sequence IDs this token belongs to (can be multiple for shared tokens)
   * @param logits Whether to compute logits for this token
   *
   * <p>Example: Add a token to sequence 0 at position 5:
   * <pre>{@code
   * batch.add(tokenId, 5, List.of(0), true);
   * }</pre>
   *
   * <p>Example: Add a shared system prompt token to multiple sequences:
   * <pre>{@code
   * batch.add(systemToken, 0, List.of(0, 1, 2, 3), false);
   * }</pre>
   *
   * <p><b>Note on performance:</b> If you're adding many tokens (100+), call
   * {@link #enableCache()} first to eliminate reflection overhead.
   */
  public void add(int token, int pos, List<Integer> seqIds, boolean logits) {
    // Use optimized version with cache for better performance
    // (this will auto-create cache on first add if not already enabled)
    BatchFieldCache cache = getOrCreateCache();
    llama_batch_add(this.segment, token, pos, seqIds, logits, cache);
  }

  /**
   * Clears all tokens from this batch, resetting it for reuse.
   */
  public void clear() {
    llama_batch_clear(this.segment);
  }

  /**
   * Decodes this batch using the given context.
   * This processes all tokens in the batch in a single forward pass.
   *
   * @return 0 on success, non-zero on error
   */
  public int decode(LlamaContext context) {
    return llama_decode(context.segment, this.segment);
  }

  /**
   * Returns the number of tokens currently in this batch.
   */
  public int nTokens() {
    return llama_batch_n_tokens(segment);
  }

  @Override
  public void free() {
    checkNotFreed();
    markFreed();
    llama_batch_free(this);
  }
}
