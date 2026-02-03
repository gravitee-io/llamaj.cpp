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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Queries GGML GPU devices for memory information.
 *
 * <p>Iterates over all registered GGML backend devices, finds GPUs
 * (type {@code GGML_BACKEND_DEVICE_TYPE_GPU} or {@code GPU_UMA}),
 * and returns raw {@code (freeBytes, totalBytes)} for the device
 * with the most free memory.
 *
 * <p>This class uses only primitive types and llama.cpp FFM bindings —
 * no external dependencies.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class GpuMemoryQuery {

  private static final int DEVICE_TYPE_GPU = 1;
  private static final int DEVICE_TYPE_GPU_UMA = 2;

  private GpuMemoryQuery() {}

  /**
   * Queries all registered GGML GPU devices and returns the memory info
   * for the device with the most free VRAM.
   *
   * @return a {@link GpuMemoryInfo} with {@code (freeBytes, totalBytes)},
   *         or {@code null} if no GPU device is found. Never throws.
   */
  public static GpuMemoryInfo queryBest() {
    try {
      long devCount = LlamaRuntime.ggml_backend_dev_count();
      GpuMemoryInfo best = null;
      try (Arena arena = Arena.ofConfined()) {
        for (long i = 0; i < devCount; i++) {
          MemorySegment dev = LlamaRuntime.ggml_backend_dev_get(i);
          int type = LlamaRuntime.ggml_backend_dev_type(dev);
          if (type == DEVICE_TYPE_GPU || type == DEVICE_TYPE_GPU_UMA) {
            long[] mem = LlamaRuntime.ggml_backend_dev_memory(arena, dev);
            if (best == null || mem[0] > best.freeBytes()) {
              best = new GpuMemoryInfo(mem[0], mem[1]);
            }
          }
        }
      }
      return best;
    } catch (Exception e) {
      return null;
    }
  }

  /**
   * Raw GPU memory info.
   *
   * @param freeBytes  free GPU memory in bytes
   * @param totalBytes total GPU memory in bytes
   */
  public record GpuMemoryInfo(long freeBytes, long totalBytes) {}
}
