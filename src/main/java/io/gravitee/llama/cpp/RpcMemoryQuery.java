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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Queries remote RPC server GPU memory without registering them as backends.
 *
 * <p>Uses {@link LlamaRuntime#ggml_backend_rpc_get_device_memory} which makes
 * a direct network call to each endpoint — no prior
 * {@link BackendRegistry#addRpcServer} registration is needed. This allows
 * pre-flight memory checks to run before the model is loaded.
 *
 * <p>When multiple RPC servers are configured (distributed inference with
 * {@code splitMode=LAYER}), llama.cpp distributes layers across devices
 * <b>proportionally to their free VRAM</b>. The total available memory is
 * therefore the <b>sum</b> of free memory across all servers, and the total
 * required memory is spread across them.
 *
 * <p>On any failure (network error, native lib not loaded, etc.) returns
 * {@code null} (never throws).
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class RpcMemoryQuery {

  private RpcMemoryQuery() {}

  /**
   * Queries GPU memory on all given RPC endpoints and returns aggregated info.
   *
   * <p>Each endpoint is queried for device 0 (the primary GPU on that server).
   * The result aggregates across all servers:
   * <ul>
   *   <li>{@code freeBytes} — sum of free memory across all servers
   *       (llama.cpp distributes layers proportionally to free VRAM)</li>
   *   <li>{@code totalBytes} — sum of total memory across all servers</li>
   * </ul>
   *
   * @param endpoints RPC server endpoints in "host:port" format.
   * @return An {@link RpcMemoryInfo} with aggregated memory, or {@code null}
   *         if the list is empty or any query fails. Never throws.
   */
  public static RpcMemoryInfo queryAll(List<String> endpoints) {
    if (endpoints == null || endpoints.isEmpty()) return null;
    try {
      return doQueryAll(endpoints);
    } catch (Exception e) {
      return null;
    }
  }

  private static RpcMemoryInfo doQueryAll(List<String> endpoints) {
    // Deduplicate endpoints — if the same host:port appears twice,
    // we must not double-count its memory. llama.cpp also deduplicates
    // at the native level (ggml_backend_rpc_add_server returns the same
    // registry entry for duplicate endpoints).
    Set<String> unique = new LinkedHashSet<>();
    for (String ep : endpoints) {
      unique.add(ep.trim());
    }

    long totalBytes = 0;
    long totalFreeBytes = 0;

    try (Arena arena = Arena.ofConfined()) {
      for (String endpoint : unique) {
        long[] mem = LlamaRuntime.ggml_backend_rpc_get_device_memory(
          arena,
          endpoint,
          0
        );
        long free = mem[0];
        long total = mem[1];
        if (total <= 0) return null; // server returned invalid data
        totalBytes += total;
        totalFreeBytes += free;
      }
    }

    if (totalFreeBytes == 0) return null;
    return new RpcMemoryInfo(totalFreeBytes, totalBytes, unique.size());
  }

  /**
   * Aggregated GPU memory across RPC servers.
   *
   * @param freeBytes    sum of free VRAM across all servers
   * @param totalBytes   sum of total VRAM across all servers
   * @param serverCount  number of RPC servers queried
   */
  public record RpcMemoryInfo(
    long freeBytes,
    long totalBytes,
    int serverCount
  ) {}
}
