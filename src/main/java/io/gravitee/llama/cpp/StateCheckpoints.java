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
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Per-sequence snapshots of the recurrent state of a hybrid or recurrent model, so prefix reuse
 * can resume close to where a new prompt diverges instead of re-reading it from token 0.
 *
 * <p>Attention layers keep one KV row per token and can be trimmed to any position. A recurrent
 * layer keeps a single running state per sequence, which llama.cpp can only
 * rewind inside the per-token snapshot window {@code n_rs_seq}
 * ({@link LlamaContextParams#nRsSeq(int)}). When a warm turn diverges farther back than that, the
 * partial trim is rejected. With a checkpoint taken at a position {@code p} at or before the
 * divergence point, the prefill restores the recurrent state at {@code p}, trims the attention
 * cache to {@code p} and decodes from there, so a warm turn costs {@code total - p} tokens.
 *
 * <p>A checkpoint holds only the recurrent and sliding-window part of the state
 * ({@code LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY}): its size is fixed by the model (it does not grow
 * with the conversation) and is logged with the first checkpoint taken by a context. The attention
 * rows of {@code [0, p)} are those the caller declared resident through
 * {@link ConversationState#initialize(String, int)}; a checkpoint is used only when its token ids
 * equal the new prompt's first {@code p} tokens and {@code p} does not exceed that declared prefix.
 *
 * <p>Checkpoints are taken by the text prefill of states that {@link
 * ConversationState#setRetainKv(boolean) retain their KV}, the only sequences a later turn can
 * resume:
 * <ul>
 *   <li>one token before the end of the prompt, a position shared both by the next turn of an
 *       agent loop (its prompt is this prompt followed by the re-rendered reply and new messages)
 *       and by an identical resend, whose reuse {@link ConversationState#initialize(String, int)}
 *       caps at the prompt length minus one;</li>
 *   <li>every {@link #interval()} prompt tokens during long prefills, when an interval is set.</li>
 * </ul>
 * Wiping a sequence at the end of a state that does not retain its KV also frees its
 * checkpoints.
 * Each sequence keeps at most {@link #maxPerSequence()} checkpoints and evicts the least recently
 * used one. Memory is bounded by {@code maxPerSequence * checkpoint size * sequences}, all of it
 * off-heap and released by {@link #clear(int)}, {@link #setMaxPerSequence(int)} or when the
 * context is freed.
 *
 * <p>Only contexts of models with a recurrent state ({@link LlamaModel#hasRecurrentState()}) take
 * checkpoints; for attention-only models this store stays empty and the prefill path is unchanged.
 * Multimodal prompts neither take nor use checkpoints. Not thread-safe, like the context.
 *
 * @author GraviteeSource Team
 */
public final class StateCheckpoints implements Freeable {

  /** Default number of checkpoints kept per sequence. */
  public static final int DEFAULT_MAX_PER_SEQUENCE = 4;

  static final int LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY = 1;
  static final int LLAMA_STATE_SEQ_FLAGS_ON_DEVICE = 2;

  private static final System.Logger LOGGER = System.getLogger(
    StateCheckpoints.class.getName()
  );

  private final LlamaContext context;
  private final boolean supported;
  private final Map<Integer, Deque<StateCheckpoint>> bySequence =
    new HashMap<>();
  private int maxPerSequence = DEFAULT_MAX_PER_SEQUENCE;
  private int interval = 0;
  private boolean onDevice = false;
  private boolean freed = false;
  private boolean sizeLogged = false;

  StateCheckpoints(LlamaContext context, boolean supported) {
    this.context = context;
    this.supported = supported;
  }

  /** Whether the context's model has a recurrent state, the only case checkpoints apply to. */
  public boolean isSupported() {
    return supported;
  }

  /** Whether checkpoints are taken and restored: a supported model and a non-zero maximum. */
  public boolean isEnabled() {
    return supported && maxPerSequence > 0 && !freed;
  }

  /** Maximum checkpoints kept per sequence; {@code 0} disables checkpoints. */
  public int maxPerSequence() {
    return maxPerSequence;
  }

  /**
   * Sets the maximum checkpoints kept per sequence, evicting the least recently used ones beyond
   * it. {@code 0} disables checkpoints and frees them all.
   *
   * @param maxPerSequence {@code >= 0}, {@value #DEFAULT_MAX_PER_SEQUENCE} by default
   * @return this store
   */
  public StateCheckpoints setMaxPerSequence(int maxPerSequence) {
    if (maxPerSequence < 0) {
      throw new IllegalArgumentException(
        "maxPerSequence must be >= 0, got %d".formatted(maxPerSequence)
      );
    }
    this.maxPerSequence = maxPerSequence;
    bySequence.values().forEach(this::evict);
    return this;
  }

  /** Prompt tokens between two checkpoints during a prefill; {@code 0} (default) disables. */
  public int interval() {
    return interval;
  }

  /**
   * Takes an extra checkpoint every {@code interval} prompt tokens during a prefill, on top of the
   * one near the end of the prompt. Useful when prompts diverge inside the previous prompt rather
   * than after it; each checkpoint counts toward {@link #maxPerSequence()}.
   *
   * @param interval {@code >= 0}, {@code 0} (default) disables
   * @return this store
   */
  public StateCheckpoints setInterval(int interval) {
    if (interval < 0) {
      throw new IllegalArgumentException(
        "interval must be >= 0, got %d".formatted(interval)
      );
    }
    this.interval = interval;
    return this;
  }

  /** Whether checkpoint data stays in backend (device) memory, see {@link #setOnDevice(boolean)}. */
  public boolean isOnDevice() {
    return onDevice;
  }

  /**
   * Keeps checkpoint data in backend memory ({@code LLAMA_STATE_SEQ_FLAGS_ON_DEVICE}): a capture
   * is a device-to-device copy into a buffer the context owns, and only the metadata goes to host
   * memory. The context holds a single such buffer per sequence and every capture overwrites it,
   * so in this mode each sequence keeps at most one checkpoint, the latest, whatever
   * {@link #maxPerSequence()} says; {@link #interval()} checkpoints then only replace one another.
   * Switching mode frees the existing checkpoints. The device buffer is released with the context.
   *
   * @param onDevice {@code true} to keep checkpoint data on the device
   * @return this store
   */
  public StateCheckpoints setOnDevice(boolean onDevice) {
    if (this.onDevice != onDevice) {
      clear();
      this.onDevice = onDevice;
    }
    return this;
  }

  /** Number of checkpoints held for a sequence. */
  public int size(int seqId) {
    var checkpoints = bySequence.get(seqId);
    return checkpoints == null ? 0 : checkpoints.size();
  }

  /** Token positions of a sequence's checkpoints, least recently used first. */
  public List<Integer> positions(int seqId) {
    var checkpoints = bySequence.get(seqId);
    return checkpoints == null
      ? List.of()
      : checkpoints.stream().map(StateCheckpoint::n).toList();
  }

  /**
   * Off-heap host bytes held by all checkpoints of this context; in {@link #setOnDevice(boolean)
   * on-device} mode only their metadata, the state itself being in backend memory.
   */
  public long byteSize() {
    return bySequence
      .values()
      .stream()
      .flatMap(Deque::stream)
      .mapToLong(StateCheckpoint::byteSize)
      .sum();
  }

  /** Frees the checkpoints of one sequence. */
  public void clear(int seqId) {
    var checkpoints = bySequence.remove(seqId);
    if (checkpoints != null) {
      checkpoints.forEach(StateCheckpoint::free);
    }
  }

  /** Frees the checkpoints of every sequence. */
  public void clear() {
    bySequence.values().forEach(d -> d.forEach(StateCheckpoint::free));
    bySequence.clear();
  }

  /**
   * Snapshots the recurrent state of {@code seqId}, which must have just consumed
   * {@code prompt[0, n)}. Skipped when disabled or when an identical checkpoint exists; a native
   * failure leaves the store unchanged.
   *
   * @return {@code true} if a checkpoint was taken
   */
  boolean capture(int seqId, int[] prompt, int n) {
    if (!isEnabled() || n <= 0 || n > prompt.length) {
      return false;
    }
    var existing = bySequence.get(seqId);
    if (
      existing != null &&
      existing.stream().anyMatch(cp -> cp.n() == n && cp.isPrefixOf(prompt))
    ) {
      return false;
    }
    MemorySegment ctx = context.getMemorySegment();
    if (onDevice) {
      clear(seqId);
    }
    long size = llama_state_seq_get_size_ext(ctx, seqId, flags());
    if (size <= 0) {
      return false;
    }
    Arena arena = Arena.ofShared();
    try {
      MemorySegment data = arena.allocate(size);
      long written = llama_state_seq_get_data_ext(
        ctx,
        data,
        size,
        seqId,
        flags()
      );
      if (written <= 0) {
        arena.close();
        return false;
      }
      logSizeOnce(seqId);
      offer(
        new StateCheckpoint(
          seqId,
          Arrays.copyOf(prompt, n),
          arena,
          data.asSlice(0, written)
        )
      );
      return true;
    } catch (RuntimeException e) {
      arena.close();
      LOGGER.log(
        System.Logger.Level.WARNING,
        "State checkpoint of sequence %d at %d failed: %s".formatted(
          seqId,
          n,
          e.getMessage()
        )
      );
      return false;
    }
  }

  /**
   * Restores the longest checkpoint of {@code seqId} that is a prefix of {@code prompt} and no
   * longer than {@code limit}, then trims the sequence to its position. The caller guarantees the
   * attention rows of {@code [0, limit)} hold {@code prompt[0, limit)}.
   *
   * @return the restored position {@code p}, so decoding resumes at {@code p}; {@code 0} when no
   *         checkpoint applies or the native restore failed, and the sequence must be wiped
   */
  int restore(int seqId, int[] prompt, int limit) {
    if (!isEnabled()) {
      return 0;
    }
    MemorySegment ctx = context.getMemorySegment();
    var memory = context.getMemory();
    for (var checkpoint : candidates(seqId, prompt, limit)) {
      int p = checkpoint.n();
      long read = llama_state_seq_set_data_ext(
        ctx,
        checkpoint.data(),
        checkpoint.data().byteSize(),
        seqId,
        flags()
      );
      if (
        read > 0 && memory.seqRm(seqId, p, -1) && memory.posMax(seqId) == p - 1
      ) {
        touch(checkpoint);
        return p;
      }
    }
    return 0;
  }

  List<StateCheckpoint> candidates(int seqId, int[] prompt, int limit) {
    var checkpoints = bySequence.get(seqId);
    if (checkpoints == null) {
      return List.of();
    }
    return checkpoints
      .stream()
      .filter(cp -> cp.n() > 0 && cp.n() <= limit && cp.isPrefixOf(prompt))
      .sorted(Comparator.comparingInt(StateCheckpoint::n).reversed())
      .toList();
  }

  void offer(StateCheckpoint checkpoint) {
    var checkpoints = bySequence.computeIfAbsent(checkpoint.seqId(), _ ->
      new ArrayDeque<>()
    );
    checkpoints.addLast(checkpoint);
    evict(checkpoints);
  }

  private void touch(StateCheckpoint checkpoint) {
    var checkpoints = bySequence.get(checkpoint.seqId());
    if (checkpoints != null && checkpoints.remove(checkpoint)) {
      checkpoints.addLast(checkpoint);
    }
  }

  private int flags() {
    return onDevice
      ? LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE
      : LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
  }

  private void evict(Deque<StateCheckpoint> checkpoints) {
    int max = onDevice ? Math.min(1, maxPerSequence) : maxPerSequence;
    while (checkpoints.size() > max) {
      checkpoints.removeFirst().free();
    }
  }

  private void logSizeOnce(int seqId) {
    if (sizeLogged) {
      return;
    }
    sizeLogged = true;
    long bytes = llama_state_seq_get_size_ext(
      context.getMemorySegment(),
      seqId,
      LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
    );
    int perSequence = onDevice ? Math.min(1, maxPerSequence) : maxPerSequence;
    String model;
    try (Arena tmp = Arena.ofConfined()) {
      model = context.getModel().desc(tmp);
    }
    LOGGER.log(
      System.Logger.Level.INFO,
      "State checkpoints for %s: %d bytes each, up to %d per sequence (%d bytes per sequence, %s)".formatted(
        model,
        bytes,
        perSequence,
        bytes * perSequence,
        onDevice ? "device memory" : "host memory"
      )
    );
  }

  @Override
  public void free() {
    freed = true;
    clear();
  }

  @Override
  public boolean isFree() {
    return freed;
  }
}
