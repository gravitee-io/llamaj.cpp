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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

/**
 * @author GraviteeSource Team
 */
class StateCheckpointsTest {

  private static final int[] PROMPT = IntStream.range(100, 200).toArray();

  private static StateCheckpoint checkpoint(int seqId, int[] tokens) {
    Arena arena = Arena.ofShared();
    return new StateCheckpoint(seqId, tokens, arena, arena.allocate(16));
  }

  private static StateCheckpoint prefix(int seqId, int n) {
    return checkpoint(seqId, Arrays.copyOf(PROMPT, n));
  }

  @Test
  void picks_the_longest_matching_prefix_within_the_limit() {
    var store = new StateCheckpoints(null, true).setMaxPerSequence(8);
    store.offer(prefix(0, 10));
    store.offer(prefix(0, 40));
    store.offer(prefix(0, 70));
    int[] diverging = PROMPT.clone();
    diverging[5] = -1;
    store.offer(checkpoint(0, Arrays.copyOf(diverging, 30)));
    store.offer(prefix(1, 60));

    assertThat(store.candidates(0, PROMPT, 50))
      .extracting(StateCheckpoint::n)
      .containsExactly(40, 10);
    assertThat(store.candidates(0, PROMPT, 70))
      .extracting(StateCheckpoint::n)
      .containsExactly(70, 40, 10);
    assertThat(store.candidates(0, diverging, 99))
      .extracting(StateCheckpoint::n)
      .containsExactly(30);
    assertThat(store.candidates(0, PROMPT, 9)).isEmpty();
    assertThat(store.candidates(2, PROMPT, 99)).isEmpty();
  }

  @Test
  void a_checkpoint_longer_than_the_prompt_never_matches() {
    var store = new StateCheckpoints(null, true);
    store.offer(prefix(0, 100));

    assertThat(store.candidates(0, Arrays.copyOf(PROMPT, 50), 99)).isEmpty();
  }

  @Test
  void evicts_the_oldest_past_the_maximum_and_frees_it() {
    var store = new StateCheckpoints(null, true).setMaxPerSequence(2);
    var first = prefix(0, 10);
    store.offer(first);
    store.offer(prefix(0, 20));
    store.offer(prefix(0, 30));
    store.offer(prefix(1, 5));

    assertThat(store.positions(0)).containsExactly(20, 30);
    assertThat(store.positions(1)).containsExactly(5);
    assertThat(first.isFree()).isTrue();
    assertThat(store.byteSize()).isEqualTo(3 * 16);
  }

  @Test
  void zero_maximum_disables_and_frees_everything() {
    var store = new StateCheckpoints(null, true);
    var held = prefix(0, 10);
    store.offer(held);
    assertThat(store.isEnabled()).isTrue();

    store.setMaxPerSequence(0);

    assertThat(store.isEnabled()).isFalse();
    assertThat(store.size(0)).isZero();
    assertThat(held.isFree()).isTrue();
    assertThat(store.capture(0, PROMPT, 10)).isFalse();
    assertThat(store.restore(0, PROMPT, 99)).isZero();
  }

  @Test
  void unsupported_models_never_enable_checkpoints() {
    var store = new StateCheckpoints(null, false);

    assertThat(store.isSupported()).isFalse();
    assertThat(store.isEnabled()).isFalse();
    assertThat(store.capture(0, PROMPT, 10)).isFalse();
    assertThat(store.restore(0, PROMPT, 99)).isZero();
    assertThat(store.byteSize()).isZero();
  }

  @Test
  void clear_and_free_release_the_native_bytes() {
    var store = new StateCheckpoints(null, true);
    var a = prefix(0, 10);
    var b = prefix(1, 10);
    store.offer(a);
    store.offer(b);

    store.clear(0);
    assertThat(a.isFree()).isTrue();
    assertThat(b.isFree()).isFalse();

    store.free();
    assertThat(b.isFree()).isTrue();
    assertThat(store.isEnabled()).isFalse();
  }

  @Test
  void on_device_keeps_only_the_latest_checkpoint_per_sequence() {
    var store = new StateCheckpoints(null, true).setOnDevice(true);
    var first = prefix(0, 10);
    store.offer(first);
    store.offer(prefix(0, 20));
    store.offer(prefix(1, 5));

    assertThat(store.isOnDevice()).isTrue();
    assertThat(store.positions(0)).containsExactly(20);
    assertThat(store.positions(1)).containsExactly(5);
    assertThat(first.isFree()).isTrue();
  }

  @Test
  void switching_mode_frees_existing_checkpoints() {
    var store = new StateCheckpoints(null, true);
    var host = prefix(0, 10);
    store.offer(host);

    store.setOnDevice(false);
    assertThat(host.isFree()).isFalse();

    store.setOnDevice(true);
    assertThat(host.isFree()).isTrue();
    assertThat(store.size(0)).isZero();

    var device = prefix(0, 10);
    store.offer(device);
    store.setOnDevice(false);
    assertThat(device.isFree()).isTrue();
    assertThat(store.isOnDevice()).isFalse();
  }

  @Test
  void on_device_with_zero_maximum_stays_disabled() {
    var store = new StateCheckpoints(null, true)
      .setOnDevice(true)
      .setMaxPerSequence(0);
    store.offer(prefix(0, 10));

    assertThat(store.isEnabled()).isFalse();
    assertThat(store.size(0)).isZero();
  }

  @Test
  void rejects_negative_settings() {
    var store = new StateCheckpoints(null, true);

    assertThatThrownBy(() -> store.setMaxPerSequence(-1)).isInstanceOf(
      IllegalArgumentException.class
    );
    assertThatThrownBy(() -> store.setInterval(-1)).isInstanceOf(
      IllegalArgumentException.class
    );
  }
}
