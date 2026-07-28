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

import io.gravitee.llama.cpp.LlamaRuntime.MethodKey;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import org.junit.jupiter.api.Test;

/**
 * Cache-key semantics for the resolved-binding cache in {@link LlamaRuntime}.
 *
 * <p>The cache is keyed partly on a {@code Class<?>[]} signature, and every call site builds a
 * <em>fresh</em> array. If equality compared those by identity — as a {@code record} would — every
 * lookup would miss, the cache would grow without bound and the reflection cost it exists to remove
 * would still be paid on every native call. Nothing at runtime would fail; it would just be slower.
 * That silent failure is what these tests pin down.
 *
 * <p>No native libraries required.
 *
 * @author GraviteeSource Team
 */
class LlamaRuntimeMethodCacheTest {

  @Test
  void distinct_but_equal_signature_arrays_are_the_same_key() {
    var first = new MethodKey(
      "llama_h",
      "llama_decode",
      new Class<?>[] { MemorySegment.class, MemorySegment.class }
    );
    var second = new MethodKey(
      "llama_h",
      "llama_decode",
      new Class<?>[] { MemorySegment.class, MemorySegment.class }
    );

    assertThat(first).isEqualTo(second);
    assertThat(first.hashCode()).isEqualTo(second.hashCode());

    // The property that actually matters: a second lookup hits rather than inserting.
    var map = new HashMap<MethodKey, String>();
    map.put(first, "resolved");
    assertThat(map).containsKey(second).hasSize(1);
    map.put(second, "resolved");
    assertThat(map).hasSize(1);
  }

  @Test
  void empty_signatures_match() {
    assertThat(
      new MethodKey("llama_h", "llama_backend_init", new Class<?>[] {})
    ).isEqualTo(
      new MethodKey("llama_h", "llama_backend_init", new Class<?>[] {})
    );
  }

  @Test
  void a_different_class_name_method_or_signature_is_a_different_key() {
    var base = new MethodKey(
      "llama_h",
      "llama_decode",
      new Class<?>[] { MemorySegment.class }
    );

    assertThat(base).isNotEqualTo(
      new MethodKey(
        "llama_model_params",
        "llama_decode",
        new Class<?>[] { MemorySegment.class }
      )
    );
    assertThat(base).isNotEqualTo(
      new MethodKey(
        "llama_h",
        "llama_encode",
        new Class<?>[] { MemorySegment.class }
      )
    );
    // Overloads differ only by signature — conflating them would dispatch to the wrong binding.
    assertThat(base).isNotEqualTo(
      new MethodKey(
        "llama_h",
        "llama_decode",
        new Class<?>[] { MemorySegment.class, int.class }
      )
    );
    assertThat(base).isNotEqualTo(
      new MethodKey("llama_h", "llama_decode", new Class<?>[] { int.class })
    );
    assertThat(base).isNotEqualTo(null);
    assertThat(base).isNotEqualTo("not a key");
  }

  @Test
  void argument_order_is_significant() {
    assertThat(
      new MethodKey(
        "llama_h",
        "f",
        new Class<?>[] { MemorySegment.class, int.class }
      )
    ).isNotEqualTo(
      new MethodKey(
        "llama_h",
        "f",
        new Class<?>[] { int.class, MemorySegment.class }
      )
    );
  }
}
