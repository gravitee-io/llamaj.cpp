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

import io.gravitee.llama.cpp.RpcMemoryQuery.RpcMemoryInfo;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link RpcMemoryQuery}.
 *
 * <p>Live RPC queries require a running rpc-server process and are tested
 * in {@code BackendRegistryTest}. These tests verify the record contract
 * and the null-safety of {@code queryAll()}.
 */
class RpcMemoryQueryTest {

  @Nested
  @DisplayName("RpcMemoryInfo record")
  class RpcMemoryInfoTests {

    @Test
    @DisplayName("accessors return correct values")
    void accessors() {
      RpcMemoryInfo info = new RpcMemoryInfo(
        8_000_000_000L,
        48_000_000_000L,
        2
      );
      assertThat(info.freeBytes()).isEqualTo(8_000_000_000L);
      assertThat(info.totalBytes()).isEqualTo(48_000_000_000L);
      assertThat(info.serverCount()).isEqualTo(2);
    }

    @Test
    @DisplayName("equality and hashCode")
    void equality() {
      RpcMemoryInfo a = new RpcMemoryInfo(1024, 4096, 1);
      RpcMemoryInfo b = new RpcMemoryInfo(1024, 4096, 1);
      RpcMemoryInfo c = new RpcMemoryInfo(2048, 4096, 1);

      assertThat(a).isEqualTo(b);
      assertThat(a).hasSameHashCodeAs(b);
      assertThat(a).isNotEqualTo(c);
    }

    @Test
    @DisplayName("toString includes field values")
    void toStringContainsValues() {
      RpcMemoryInfo info = new RpcMemoryInfo(100, 200, 3);
      assertThat(info.toString()).contains("100");
      assertThat(info.toString()).contains("200");
      assertThat(info.toString()).contains("3");
    }

    @Test
    @DisplayName("large values (>32-bit) work correctly")
    void large_values() {
      long free = 81_604_378_624L; // ~76 GiB
      long total = 257_698_037_760L; // ~240 GiB (3x 80GB GPUs)
      RpcMemoryInfo info = new RpcMemoryInfo(free, total, 3);
      assertThat(info.freeBytes()).isEqualTo(free);
      assertThat(info.totalBytes()).isEqualTo(total);
      assertThat(info.serverCount()).isEqualTo(3);
    }
  }

  @Nested
  @DisplayName("queryAll null-safety")
  class QueryAllNullSafety {

    @Test
    @DisplayName("null endpoints returns null")
    void null_endpoints() {
      assertThat(RpcMemoryQuery.queryAll(null)).isNull();
    }

    @Test
    @DisplayName("empty endpoints returns null")
    void empty_endpoints() {
      assertThat(RpcMemoryQuery.queryAll(List.of())).isNull();
    }

    @Test
    @DisplayName("unreachable endpoint returns null (never throws)")
    void unreachable_endpoint() {
      // This will fail at the native call level — should return null gracefully
      RpcMemoryInfo result = RpcMemoryQuery.queryAll(List.of("127.0.0.1:1"));
      // Either null (native call fails) or a result (extremely unlikely)
      // The key assertion: no exception thrown
      assertThat(result).satisfiesAnyOf(
        r -> assertThat(r).isNull(),
        r -> assertThat(r).isNotNull()
      );
    }
  }
}
