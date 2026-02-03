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

import io.gravitee.llama.cpp.GpuMemoryQuery.GpuMemoryInfo;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class GpuMemoryQueryTest {

  @Test
  @DisplayName("GpuMemoryInfo record: accessors return correct values")
  void gpuMemoryInfo_accessors() {
    GpuMemoryInfo info = new GpuMemoryInfo(8_000_000_000L, 24_000_000_000L);
    assertThat(info.freeBytes()).isEqualTo(8_000_000_000L);
    assertThat(info.totalBytes()).isEqualTo(24_000_000_000L);
  }

  @Test
  @DisplayName("GpuMemoryInfo record: equality and hashCode")
  void gpuMemoryInfo_equality() {
    GpuMemoryInfo a = new GpuMemoryInfo(1024, 4096);
    GpuMemoryInfo b = new GpuMemoryInfo(1024, 4096);
    GpuMemoryInfo c = new GpuMemoryInfo(2048, 4096);

    assertThat(a).isEqualTo(b);
    assertThat(a).hasSameHashCodeAs(b);
    assertThat(a).isNotEqualTo(c);
  }

  @Test
  @DisplayName("GpuMemoryInfo record: toString includes field values")
  void gpuMemoryInfo_toString() {
    GpuMemoryInfo info = new GpuMemoryInfo(100, 200);
    assertThat(info.toString()).contains("100");
    assertThat(info.toString()).contains("200");
  }

  @Test
  @DisplayName("GpuMemoryInfo record: zero values are valid")
  void gpuMemoryInfo_zero_values() {
    GpuMemoryInfo info = new GpuMemoryInfo(0, 0);
    assertThat(info.freeBytes()).isEqualTo(0);
    assertThat(info.totalBytes()).isEqualTo(0);
  }

  @Test
  @DisplayName("GpuMemoryInfo record: large values (>32-bit) work correctly")
  void gpuMemoryInfo_large_values() {
    long free = 81_604_378_624L;
    long total = 85_899_345_920L;
    GpuMemoryInfo info = new GpuMemoryInfo(free, total);
    assertThat(info.freeBytes()).isEqualTo(free);
    assertThat(info.totalBytes()).isEqualTo(total);
  }
}
