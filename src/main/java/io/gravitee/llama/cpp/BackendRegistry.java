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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Provides a high-level Java API for querying and managing GGML backends.
 * <p>
 * This class wraps the low-level {@link LlamaRuntime} backend functions to provide
 * convenient access to backend discovery, device enumeration, and RPC server registration.
 * <p>
 * Must be used after {@link LlamaRuntime#llama_backend_init()} and
 * {@link LlamaRuntime#ggml_backend_load_all_from_path(Arena, String)} have been called.
 *
 * <h3>Usage example:</h3>
 * <pre>{@code
 * // After backend initialization
 * LlamaRuntime.llama_backend_init();
 * LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);
 *
 * // Discover backends and devices
 * var backends = BackendRegistry.listBackends();
 * var devices = BackendRegistry.listDevices();
 *
 * // Add RPC servers for distributed inference
 * if (BackendRegistry.supportsRpc()) {
 *     BackendRegistry.addRpcServer(arena, "192.168.1.10:50052");
 *     BackendRegistry.addRpcServer(arena, "192.168.1.11:50052");
 * }
 * }</pre>
 *
 * @author Remi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class BackendRegistry {

  private BackendRegistry() {}

  /**
   * Lists all registered backends.
   *
   * @return An unmodifiable list of {@link GgmlBackendInfo} entries.
   */
  public static List<GgmlBackendInfo> listBackends() {
    long count = LlamaRuntime.ggml_backend_reg_count();
    var backends = new ArrayList<GgmlBackendInfo>((int) count);
    for (long i = 0; i < count; i++) {
      MemorySegment reg = LlamaRuntime.ggml_backend_reg_get(i);
      MemorySegment namePtr = LlamaRuntime.ggml_backend_reg_name(reg);
      String name = namePtr.getString(0);
      backends.add(new GgmlBackendInfo(name, i));
    }
    return Collections.unmodifiableList(backends);
  }

  /**
   * Lists all available devices across all registered backends.
   *
   * @return An unmodifiable list of {@link GgmlDeviceInfo} entries.
   */
  public static List<GgmlDeviceInfo> listDevices() {
    long count = LlamaRuntime.ggml_backend_dev_count();
    var devices = new ArrayList<GgmlDeviceInfo>((int) count);
    for (long i = 0; i < count; i++) {
      MemorySegment dev = LlamaRuntime.ggml_backend_dev_get(i);
      String name = LlamaRuntime.ggml_backend_dev_name(dev).getString(0);
      String description = LlamaRuntime.ggml_backend_dev_description(
        dev
      ).getString(0);

      // Resolve the parent backend name
      MemorySegment reg = LlamaRuntime.ggml_backend_dev_backend_reg(dev);
      String backendName = LlamaRuntime.ggml_backend_reg_name(reg).getString(0);

      devices.add(new GgmlDeviceInfo(name, description, i, backendName));
    }
    return Collections.unmodifiableList(devices);
  }

  /**
   * Returns the raw device handles for all devices whose backend name is "RPC".
   * These handles can be passed to {@link LlamaModelParams#devices(java.lang.foreign.Arena, List)}
   * to restrict offloading to only RPC devices.
   * <p>
   * Note: RPC devices registered via {@link #addRpcServer} are NOT in the global device
   * registry. Use {@link #addRpcServer} which returns the device handles directly, or
   * use {@link #addRpcServers} which collects them.
   *
   * @return A list of MemorySegment device handles for RPC devices.
   */
  public static List<MemorySegment> getRpcDeviceHandles() {
    // Check global registry first (for statically-registered RPC devices)
    long count = LlamaRuntime.ggml_backend_dev_count();
    var rpcDevices = new ArrayList<MemorySegment>();
    for (long i = 0; i < count; i++) {
      MemorySegment dev = LlamaRuntime.ggml_backend_dev_get(i);
      MemorySegment reg = LlamaRuntime.ggml_backend_dev_backend_reg(dev);
      String backendName = LlamaRuntime.ggml_backend_reg_name(reg).getString(0);
      if ("RPC".equals(backendName)) {
        rpcDevices.add(dev);
      }
    }
    // Also include devices from dynamically-added RPC servers
    rpcDevices.addAll(rpcDeviceHandles);
    return rpcDevices;
  }

  // Accumulates device handles from addRpcServer calls
  private static final List<MemorySegment> rpcDeviceHandles = new ArrayList<>();

  /**
   * Checks if RPC backend support is available in the loaded llama.cpp library.
   *
   * @return true if the RPC backend plugin was loaded and RPC is supported.
   */
  public static boolean supportsRpc() {
    return LlamaRuntime.llama_supports_rpc();
  }

  /**
   * Loads a single backend from a specific shared library path.
   * <p>
   * Use this instead of {@link LlamaRuntime#ggml_backend_load_all_from_path(Arena, String)}
   * when you want to load only specific backends rather than everything in a directory.
   * <p>
   * Must be called after {@link LlamaRuntime#llama_backend_init()}.
   *
   * @param arena The Arena for memory allocation.
   * @param path  Absolute path to the backend shared library
   *              (e.g., "/path/to/libggml-metal.dylib" or "/path/to/libggml-cpu.so").
   * @return A {@link GgmlBackendInfo} for the loaded backend.
   * @throws LlamaException if the backend could not be loaded.
   */
  public static GgmlBackendInfo loadBackend(Arena arena, String path) {
    MemorySegment reg = LlamaRuntime.ggml_backend_load(arena, path);
    if (reg == null || reg.equals(MemorySegment.NULL)) {
      throw new LlamaException("Failed to load backend from: " + path);
    }
    String name = LlamaRuntime.ggml_backend_reg_name(reg).getString(0);
    long index = LlamaRuntime.ggml_backend_reg_count() - 1;
    return new GgmlBackendInfo(name, index);
  }

  /**
   * Registers a remote RPC server as a backend device.
   * <p>
   * The returned devices are NOT added to the global device registry.
   * They must be passed explicitly to {@link LlamaModelParams#devices(Arena, List)}
   * for the model to use them.
   * <p>
   * <b>Important:</b> The remote rpc-server must be the same llama.cpp version (b7943)
   * as the client library. A version mismatch will crash the process (the native RPC
   * client calls abort() on protocol errors).
   * <p>
   * A TCP connectivity check is performed before calling into native code to avoid
   * unrecoverable crashes from unreachable servers.
   *
   * @param arena    The Arena for memory allocation.
   * @param endpoint The RPC server endpoint in "host:port" format (e.g., "192.168.1.10:50052").
   * @return A list of device handles from the RPC server.
   * @throws IllegalStateException if RPC is not supported.
   * @throws LlamaException if the RPC server could not be reached.
   */
  public static List<MemorySegment> addRpcServer(Arena arena, String endpoint) {
    if (!supportsRpc()) {
      throw new IllegalStateException(
        "RPC backend is not supported. Ensure libggml-rpc is present in the library path."
      );
    }
    // Pre-flight TCP check to avoid native abort() on unreachable servers
    checkRpcConnectivity(endpoint);

    MemorySegment reg = LlamaRuntime.ggml_backend_rpc_add_server(
      arena,
      endpoint
    );
    if (reg == null || reg.equals(MemorySegment.NULL)) {
      throw new LlamaException("Failed to connect to RPC server: " + endpoint);
    }
    // Extract devices from the returned registry entry
    long devCount = LlamaRuntime.ggml_backend_reg_dev_count(reg);
    var devices = new ArrayList<MemorySegment>((int) devCount);
    for (long i = 0; i < devCount; i++) {
      MemorySegment dev = LlamaRuntime.ggml_backend_reg_dev_get(reg, i);
      devices.add(dev);
      rpcDeviceHandles.add(dev);
    }
    return devices;
  }

  /**
   * Registers multiple remote RPC servers as backend devices.
   * <p>
   * Duplicate endpoints are ignored — each unique {@code host:port} is registered
   * only once. This prevents double-connecting to the same server when the
   * configuration contains duplicates.
   *
   * @param arena     The Arena for memory allocation.
   * @param endpoints The RPC server endpoints in "host:port" format.
   * @throws IllegalStateException if RPC is not supported.
   */
  public static void addRpcServers(Arena arena, String... endpoints) {
    Set<String> seen = new LinkedHashSet<>();
    for (String endpoint : endpoints) {
      String trimmed = endpoint.trim();
      if (seen.add(trimmed)) {
        addRpcServer(arena, trimmed);
      }
    }
  }

  /**
   * Queries the memory available on a remote RPC device.
   *
   * @param arena    The Arena for memory allocation.
   * @param endpoint The RPC server endpoint in "host:port" format.
   * @param device   The device index on the remote server (usually 0).
   * @return An {@link RpcDeviceMemory} record with free and total memory in bytes.
   */
  public static RpcDeviceMemory queryRpcMemory(
    Arena arena,
    String endpoint,
    int device
  ) {
    long[] memory = LlamaRuntime.ggml_backend_rpc_get_device_memory(
      arena,
      endpoint,
      device
    );
    return new RpcDeviceMemory(memory[0], memory[1]);
  }

  /**
   * Memory information for a remote RPC device.
   *
   * @param freeBytes  Available memory in bytes.
   * @param totalBytes Total memory in bytes.
   */
  public record RpcDeviceMemory(long freeBytes, long totalBytes) {
    /**
     * @return The used memory in bytes.
     */
    public long usedBytes() {
      return totalBytes - freeBytes;
    }

    @Override
    public String toString() {
      return "RpcDeviceMemory[free=%s, total=%s, used=%s]".formatted(
        humanReadable(freeBytes),
        humanReadable(totalBytes),
        humanReadable(usedBytes())
      );
    }

    private static String humanReadable(long bytes) {
      if (bytes < 1024) return bytes + " B";
      double value = bytes;
      String[] units = { "B", "KiB", "MiB", "GiB", "TiB" };
      int unit = 0;
      while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit++;
      }
      return "%.1f %s".formatted(value, units[unit]);
    }
  }

  /**
   * Prints a summary of all registered backends and devices to stdout.
   * Useful for diagnostics during startup.
   */
  public static void printSummary() {
    var backends = listBackends();
    var devices = listDevices();

    System.out.println("Registered backends (" + backends.size() + "):");
    for (var backend : backends) {
      System.out.println("  [" + backend.index() + "] " + backend.name());
    }

    System.out.println("Available devices (" + devices.size() + "):");
    for (var device : devices) {
      System.out.println(
        "  [" +
          device.index() +
          "] " +
          device.name() +
          " (" +
          device.description() +
          ") - backend: " +
          device.backendName()
      );
    }

    System.out.println("RPC support: " + (supportsRpc() ? "yes" : "no"));
    System.out.println(
      "GPU offload: " +
        (LlamaRuntime.llama_supports_gpu_offload() ? "yes" : "no")
    );
  }

  private static final int RPC_CONNECT_TIMEOUT_MS = 5000;

  /**
   * Performs a TCP connectivity check to the RPC server before calling native code.
   * This prevents unrecoverable process crashes from GGML_ABORT when the native
   * RPC client can't complete the protocol handshake.
   *
   * @param endpoint The "host:port" endpoint string.
   * @throws LlamaException if the server is unreachable.
   */
  private static void checkRpcConnectivity(String endpoint) {
    String[] parts = endpoint.split(":");
    if (parts.length != 2) {
      throw new LlamaException(
        "Invalid RPC endpoint format (expected host:port): " + endpoint
      );
    }
    String host = parts[0];
    int port;
    try {
      port = Integer.parseInt(parts[1]);
    } catch (NumberFormatException e) {
      throw new LlamaException("Invalid port in RPC endpoint: " + endpoint);
    }
    try (var socket = new Socket()) {
      socket.connect(new InetSocketAddress(host, port), RPC_CONNECT_TIMEOUT_MS);
    } catch (IOException e) {
      throw new LlamaException(
        "Cannot reach RPC server at %s: %s".formatted(endpoint, e.getMessage())
      );
    }
  }
}
