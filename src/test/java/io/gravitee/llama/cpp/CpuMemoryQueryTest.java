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

import io.gravitee.llama.cpp.CpuMemoryQuery.CpuMemoryInfo;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link CpuMemoryQuery}.
 *
 * <p>The implementation uses {@link com.sun.management.OperatingSystemMXBean}
 * which works on all standard JDK distributions. Tests verify the record
 * contract and the live query on Linux/macOS.
 */
class CpuMemoryQueryTest {

  @Nested
  @DisplayName("CpuMemoryInfo record")
  class CpuMemoryInfoTests {

    @Test
    @DisplayName("accessors return correct values")
    void accessors() {
      CpuMemoryInfo info = new CpuMemoryInfo(16_000_000_000L, 32_000_000_000L);
      assertThat(info.freeBytes()).isEqualTo(16_000_000_000L);
      assertThat(info.totalBytes()).isEqualTo(32_000_000_000L);
    }

    @Test
    @DisplayName("equality and hashCode")
    void equality() {
      CpuMemoryInfo a = new CpuMemoryInfo(1024, 4096);
      CpuMemoryInfo b = new CpuMemoryInfo(1024, 4096);
      CpuMemoryInfo c = new CpuMemoryInfo(2048, 4096);

      assertThat(a).isEqualTo(b);
      assertThat(a).hasSameHashCodeAs(b);
      assertThat(a).isNotEqualTo(c);
    }

    @Test
    @DisplayName("toString includes field values")
    void toStringContainsValues() {
      CpuMemoryInfo info = new CpuMemoryInfo(100, 200);
      assertThat(info.toString()).contains("100");
      assertThat(info.toString()).contains("200");
    }

    @Test
    @DisplayName("large values (>32-bit) work correctly")
    void large_values() {
      long free = 123_456_789_012L;
      long total = 274_877_906_944L; // 256 GiB
      CpuMemoryInfo info = new CpuMemoryInfo(free, total);
      assertThat(info.freeBytes()).isEqualTo(free);
      assertThat(info.totalBytes()).isEqualTo(total);
    }
  }

  @Nested
  @DisplayName("Live query via OperatingSystemMXBean")
  class LiveQuery {

    @Test
    @DisplayName("query() returns non-null with valid values")
    void live_query_returns_result() {
      CpuMemoryInfo info = CpuMemoryQuery.query();

      assertThat(info).isNotNull();
      assertThat(info.totalBytes()).isGreaterThan(0);
      assertThat(info.freeBytes()).isGreaterThanOrEqualTo(0);
      assertThat(info.freeBytes()).isLessThanOrEqualTo(info.totalBytes());
    }

    @Test
    @DisplayName("total memory is at least 1 GiB (sanity check)")
    void total_memory_sanity() {
      CpuMemoryInfo info = CpuMemoryQuery.query();

      assertThat(info).isNotNull();
      long oneGiB = 1024L * 1024L * 1024L;
      assertThat(info.totalBytes()).isGreaterThanOrEqualTo(oneGiB);
    }

    @Test
    @DisplayName("consecutive queries return consistent total memory")
    void consecutive_queries_consistent_total() {
      CpuMemoryInfo first = CpuMemoryQuery.query();
      CpuMemoryInfo second = CpuMemoryQuery.query();

      assertThat(first).isNotNull();
      assertThat(second).isNotNull();
      // Total physical RAM doesn't change between calls
      assertThat(first.totalBytes()).isEqualTo(second.totalBytes());
    }
  }
}
