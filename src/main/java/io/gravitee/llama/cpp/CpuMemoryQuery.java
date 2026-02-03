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

import java.lang.management.ManagementFactory;

/**
 * Queries system CPU (RAM) memory via the JDK management API.
 *
 * <p>Provides the same contract as {@link GpuMemoryQuery}: a static
 * {@link #query()} method returning a nullable record with
 * {@code (freeBytes, totalBytes)}. This allows {@code LlamaMemoryEstimator}
 * to fall back to CPU memory when no GPU is available or when
 * {@code nGpuLayers=0}.
 *
 * <p>Uses {@link com.sun.management.OperatingSystemMXBean} which is available
 * on all standard JDK distributions (OpenJDK, Oracle, GraalVM, etc.) and
 * works on Linux, macOS, and Windows. No process spawning, no file parsing,
 * no FFM bindings.
 *
 * <p>On any failure returns {@code null} (never throws).
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class CpuMemoryQuery {

  private CpuMemoryQuery() {}

  /**
   * Queries system RAM and returns available + total memory.
   *
   * @return a {@link CpuMemoryInfo} with {@code (freeBytes, totalBytes)},
   *         or {@code null} if the query fails. Never throws.
   */
  public static CpuMemoryInfo query() {
    try {
      var osBean =
        (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
      long total = osBean.getTotalMemorySize();
      long free = osBean.getFreeMemorySize();
      if (total <= 0 || free < 0) return null;
      return new CpuMemoryInfo(free, total);
    } catch (Exception e) {
      return null;
    }
  }

  /**
   * Raw CPU (system RAM) memory info.
   *
   * @param freeBytes  free system memory in bytes
   * @param totalBytes total physical memory in bytes
   */
  public record CpuMemoryInfo(long freeBytes, long totalBytes) {}
}
