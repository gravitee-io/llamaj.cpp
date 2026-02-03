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

import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_TO_DOWNLOAD;
import static io.gravitee.llama.cpp.LlamaCppTest.SYSTEM;
import static io.gravitee.llama.cpp.LlamaCppTest.buildMessages;
import static io.gravitee.llama.cpp.LlamaCppTest.getModelPath;
import static io.gravitee.llama.cpp.LlamaCppTest.getPrompt;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assumptions.assumeThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

/**
 * Tests for {@link BackendRegistry}, {@link GgmlBackendInfo}, and {@link GgmlDeviceInfo}.
 *
 * <p>The basic tests (backend/device listing, RPC support check) run against the local
 * machine's native backends. The RPC integration test automatically starts a local
 * rpc-server process if the binary is available at {@code ~/.llama.cpp/rpc-server/rpc-server},
 * and tears it down after the test completes.
 *
 * @author Remi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class BackendRegistryTest {

  private static final String RPC_HOST = "127.0.0.1";
  private static final int RPC_PORT = 50099; // Use a non-standard port to avoid conflicts
  private static final String RPC_ENDPOINT = RPC_HOST + ":" + RPC_PORT;
  private static final Path RPC_SERVER_BINARY = Path.of(
    System.getProperty("user.home"),
    ".llama.cpp",
    "rpc-server",
    "rpc-server"
  );
  private static final int STARTUP_TIMEOUT_MS = 10_000;
  private static final int POLL_INTERVAL_MS = 200;

  private static Arena arena;
  private static Process rpcServerProcess;

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();

    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

    // Start the RPC server if the binary exists
    if (Files.isExecutable(RPC_SERVER_BINARY)) {
      rpcServerProcess = startRpcServer();
    }
  }

  @AfterAll
  static void afterAll() {
    stopRpcServer();
    LlamaRuntime.llama_backend_free();
    arena.close();
  }

  @Test
  @Order(1)
  void should_list_at_least_one_backend() {
    List<GgmlBackendInfo> backends = BackendRegistry.listBackends();

    assertThat(backends).isNotEmpty();
    assertThat(backends).anyMatch(b -> b.name().toLowerCase().contains("cpu"));

    for (var backend : backends) {
      assertThat(backend.name()).isNotNull().isNotBlank();
      assertThat(backend.index()).isGreaterThanOrEqualTo(0);
    }
  }

  @Test
  @Order(2)
  void should_list_at_least_one_device() {
    List<GgmlDeviceInfo> devices = BackendRegistry.listDevices();

    assertThat(devices).isNotEmpty();

    for (var device : devices) {
      assertThat(device.name()).isNotNull().isNotBlank();
      assertThat(device.description()).isNotNull().isNotBlank();
      assertThat(device.index()).isGreaterThanOrEqualTo(0);
      assertThat(device.backendName()).isNotNull().isNotBlank();
    }
  }

  @Test
  @Order(3)
  void should_report_rpc_support() {
    boolean rpcSupported = BackendRegistry.supportsRpc();
    assertThat(rpcSupported).isTrue();
  }

  @Test
  @Order(4)
  void should_print_summary_without_error() {
    BackendRegistry.printSummary();
  }

  @Test
  @Order(5)
  void backend_info_to_string_should_include_name() {
    var info = new GgmlBackendInfo("CPU", 0);
    assertThat(info.toString()).contains("CPU").contains("0");
  }

  @Test
  @Order(6)
  void device_info_to_string_should_include_all_fields() {
    var info = new GgmlDeviceInfo("Metal", "Apple M2 Pro", 1, "Metal");
    assertThat(info.toString())
      .contains("Metal")
      .contains("Apple M2 Pro")
      .contains("1");
  }

  @Test
  @Order(7)
  void rpc_device_memory_to_string_should_be_human_readable() {
    var memory = new BackendRegistry.RpcDeviceMemory(
      4L * 1024 * 1024 * 1024,
      8L * 1024 * 1024 * 1024
    );
    assertThat(memory.freeBytes()).isEqualTo(4L * 1024 * 1024 * 1024);
    assertThat(memory.totalBytes()).isEqualTo(8L * 1024 * 1024 * 1024);
    assertThat(memory.usedBytes()).isEqualTo(4L * 1024 * 1024 * 1024);
    assertThat(memory.toString()).contains("GiB");
  }

  /**
   * Integration test that starts a local rpc-server, connects to it,
   * registers it as a backend, and queries its device memory.
   * Skipped if the rpc-server binary is not available.
   */
  @Test
  @Order(8)
  void should_register_rpc_server_and_query_memory() {
    assumeThat(rpcServerProcess)
      .as("rpc-server binary not found at %s — skipping", RPC_SERVER_BINARY)
      .isNotNull();
    assumeThat(rpcServerProcess.isAlive())
      .as("rpc-server process is not running — skipping")
      .isTrue();

    // Register the RPC server — returns device handles directly
    // (RPC devices are NOT added to the global device registry)
    var rpcDevices = BackendRegistry.addRpcServer(arena, RPC_ENDPOINT);
    assertThat(rpcDevices).isNotEmpty();

    // The device handles should also be tracked internally
    var allRpcHandles = BackendRegistry.getRpcDeviceHandles();
    assertThat(allRpcHandles).isNotEmpty();

    // Query memory on the remote device
    var memory = BackendRegistry.queryRpcMemory(arena, RPC_ENDPOINT, 0);
    assertThat(memory.totalBytes()).isGreaterThan(0);
    assertThat(memory.freeBytes()).isGreaterThanOrEqualTo(0);
    assertThat(memory.freeBytes()).isLessThanOrEqualTo(memory.totalBytes());

    System.out.println("RPC device memory: " + memory);

    // Print updated summary
    BackendRegistry.printSummary();
  }

  /**
   * End-to-end inference test over RPC. Loads a model with weights offloaded
   * exclusively to the local rpc-server, runs a simple prompt, and verifies
   * tokens are generated.
   * Skipped if the rpc-server binary is not available.
   */
  @Test
  @Order(9)
  void should_perform_inference_over_rpc() {
    assumeThat(rpcServerProcess)
      .as("rpc-server not available — skipping")
      .isNotNull();
    assumeThat(rpcServerProcess.isAlive())
      .as("rpc-server not running — skipping")
      .isTrue();

    Path modelPath = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);

    // Reuse device handles registered in the previous test
    var rpcDevices = BackendRegistry.getRpcDeviceHandles();
    assertThat(rpcDevices).isNotEmpty();

    // Load model with RPC offloading only (no local GPU)
    var modelParams = new LlamaModelParams(arena)
      .nGpuLayers(99)
      .useMlock(true)
      .useMmap(true);
    modelParams.devices(arena, rpcDevices);

    var model = new LlamaModel(arena, modelPath, modelParams);
    var contextParams = new LlamaContextParams(arena).noPerf(false);
    var context = new LlamaContext(arena, model, contextParams);
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);
    var sampler = new LlamaSampler(arena).seed(new Random().nextInt());

    var prompt = getPrompt(
      model,
      arena,
      buildMessages(arena, SYSTEM, "What is the capital of France?"),
      contextParams
    );

    var state = ConversationState.create(
      arena,
      context,
      tokenizer,
      sampler
    ).initialize(prompt);
    var iterator = new DefaultLlamaIterator(state);

    String output = iterator
      .stream()
      .reduce(LlamaOutput::merge)
      .orElse(new LlamaOutput("", 0))
      .content();

    System.out.println("RPC inference output: " + output);
    assertThat(state.getInputTokens()).isGreaterThan(0);
    assertThat(state.getAnswerTokens()).isGreaterThan(0);
    assertThat(output).isNotBlank();
    assertThat(state.getFinishReason()).isIn(
      FinishReason.EOS,
      FinishReason.LENGTH,
      FinishReason.STOP
    );

    context.free();
    sampler.free();
    model.free();
  }

  // ---- RPC server lifecycle ----

  private static Process startRpcServer() {
    try {
      // Use MTL0 on macOS to avoid BLAS backend crashes on unsupported ops
      // Use CPU on Linux (no GPU assumed in CI)
      String os = System.getProperty("os.name", "").toLowerCase();
      boolean isMac = os.contains("mac");

      var command = new java.util.ArrayList<>(
        List.of(
          RPC_SERVER_BINARY.toString(),
          "-H",
          RPC_HOST,
          "-p",
          String.valueOf(RPC_PORT)
        )
      );
      if (isMac) {
        command.add("-d");
        command.add("MTL0");
      } else {
        command.add("-d");
        command.add("CPU");
      }

      var processBuilder = new ProcessBuilder(command);
      processBuilder.directory(RPC_SERVER_BINARY.getParent().toFile());
      processBuilder.inheritIO();

      Process process = processBuilder.start();

      // Wait for the server to be ready
      if (!waitForPort(RPC_HOST, RPC_PORT, STARTUP_TIMEOUT_MS)) {
        process.destroyForcibly();
        System.err.println(
          "rpc-server failed to start within " + STARTUP_TIMEOUT_MS + "ms"
        );
        return null;
      }

      System.out.println(
        "rpc-server started on " + RPC_ENDPOINT + " (pid=" + process.pid() + ")"
      );
      return process;
    } catch (IOException e) {
      System.err.println("Failed to start rpc-server: " + e.getMessage());
      return null;
    }
  }

  private static void stopRpcServer() {
    if (rpcServerProcess != null && rpcServerProcess.isAlive()) {
      System.out.println(
        "Stopping rpc-server (pid=" + rpcServerProcess.pid() + ")..."
      );
      rpcServerProcess.destroy();
      try {
        boolean exited = rpcServerProcess.waitFor(
          5,
          java.util.concurrent.TimeUnit.SECONDS
        );
        if (!exited) {
          rpcServerProcess.destroyForcibly();
        }
      } catch (InterruptedException e) {
        rpcServerProcess.destroyForcibly();
        Thread.currentThread().interrupt();
      }
    }
  }

  private static boolean waitForPort(String host, int port, int timeoutMs) {
    long deadline = System.currentTimeMillis() + timeoutMs;
    while (System.currentTimeMillis() < deadline) {
      try (var socket = new Socket()) {
        socket.connect(new InetSocketAddress(host, port), 500);
        return true;
      } catch (IOException e) {
        // Not ready yet
      }
      try {
        Thread.sleep(POLL_INTERVAL_MS);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        return false;
      }
    }
    return false;
  }
}
