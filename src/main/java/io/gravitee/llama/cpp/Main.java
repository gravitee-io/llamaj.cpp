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

import static io.gravitee.llama.cpp.LlamaRuntime.ggml_backend_load_all;
import static java.lang.Boolean.parseBoolean;
import static java.lang.Float.parseFloat;
import static java.lang.Integer.parseInt;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {

  public static final Arena ARENA = Arena.ofAuto();

  public static void main(String[] args) {
    Map<String, String> params = parseArgs(args);

    String modelGguf = params.get("model");
    String systemMessage = params.getOrDefault("system", "You are a helpful AI assistant.");

    int nGpuLayers = parseInt(params.getOrDefault("n_gpu_layers", "999"));
    boolean useMlock = parseBoolean(params.getOrDefault("use_mlock", "true"));
    boolean useMmap = parseBoolean(params.getOrDefault("use_mmap", "true"));

    float temperature = parseFloat(params.getOrDefault("temperature", "0.4f"));
    float minP = parseFloat(params.getOrDefault("min_p", "0.1f"));
    int minPWindow = parseInt(params.getOrDefault("min_p_window", "40"));
    int topK = parseInt(params.getOrDefault("top_k", "10"));
    float topP = parseFloat(params.getOrDefault("top_p", "0.2f"));
    int topPWindow = parseInt(params.getOrDefault("top_p_window", "10"));
    int seed = parseInt(params.getOrDefault("seed", String.valueOf(new Random().nextInt())));

    int nCtx = parseInt(params.getOrDefault("n_ctx", "512"));
    int nBatch = parseInt(params.getOrDefault("n_batch", "512"));
    int nSeqMax = parseInt(params.getOrDefault("n_seq_max", "512"));
    int quota = parseInt(params.getOrDefault("quota", "512"));

    String logLevelStr = params.getOrDefault("log_level", "ERROR").toUpperCase();
    LlamaLogLevel logLevel;
    try {
      logLevel = LlamaLogLevel.valueOf(logLevelStr);
    } catch (IllegalArgumentException e) {
      System.err.println("Invalid log level: %s. Using default: ERROR.".formatted(logLevelStr));
      logLevel = LlamaLogLevel.ERROR;
    }

    if (modelGguf == null) {
      System.err.println("Usage: java -jar your-app.jar --model <path_to_gguf_model> [options...]");
      System.err.println("Options:");
      System.err.println("  --system <message>       : System message (default: \"You are a helpful AI assistant.\")");
      System.err.println("  --n_gpu_layers <int>     : Number of GPU layers (default: 999)");
      System.err.println("  --use_mlock <boolean>    : Use mlock (default: true)");
      System.err.println("  --use_mmap <boolean>     : Use mmap (default: true)");
      System.err.println("  --temperature <float>    : Sampler temperature (default: 0.4)");
      System.err.println("  --min_p <float>          : Sampler min_p (default: 0.1)");
      System.err.println("  --min_p_window <int>     : Sampler min_p_window (default: 40)");
      System.err.println("  --top_k <int>            : Sampler top_k (default: 10)");
      System.err.println("  --top_p <float>          : Sampler top_p (default: 0.2)");
      System.err.println("  --top_p_window <int>     : Sampler top_p_window (default: 10)");
      System.err.println("  --seed <long>            : Sampler seed (default: random)");
      System.err.println("  --n_ctx <int>            : Context size (default: 512)");
      System.err.println("  --n_batch <int>          : Batch size (default: 512)");
      System.err.println("  --n_seq_max <int>        : Max sequence length (default: 512)");
      System.err.println("  --quota <int>            : Iterator quota (default: 512)");
      System.err.println("  --log_level <level>      : Logging level (ERROR, WARN, INFO, DEBUG, default: ERROR)"); // New option in usage
      System.exit(1);
    }

    LlamaLibLoader.load();
    ggml_backend_load_all();

    var logger = new LlamaLogger(ARENA);
    logger.setLogging(logLevel); // Applying the parsed log level

    var modelParameters = new LlamaModelParams(ARENA);
    modelParameters.nGpuLayers(nGpuLayers).useMlock(useMlock).useMmap(useMmap);

    LlamaModel model = new LlamaModel(ARENA, Path.of(modelGguf).toAbsolutePath(), modelParameters);
    LlamaVocab vocab = new LlamaVocab(model);

    LlamaSampler sampler = new LlamaSampler(ARENA)
      .temperature(temperature)
      .minP(minP, minPWindow)
      .topK(topK)
      .topP(topP, topPWindow)
      .seed(seed);

    var contextParams = new LlamaContextParams(ARENA).nCtx(nCtx).nBatch(nBatch).nSeqMax(nSeqMax);

    String input = "";
    while (!input.trim().equals("bye")) {
      Scanner scanIn = new Scanner(System.in);
      System.out.print("Please enter your prompt: ");
      input = scanIn.nextLine();

      if (input.isBlank()) {
        break;
      }

      String prompt = buildPrompt(model, systemMessage, input, contextParams);

      LlamaIterator iterator = new SimpleLlamaIterator(ARENA, model, contextParams, vocab, sampler)
        .setQuota(quota)
        .initialize(prompt);

      iterator.stream().map(LlamaOutput::content).forEach(System.out::print);

      System.out.println();
    }

    sampler.free();
    model.free();
  }

  private static String buildPrompt(LlamaModel model, String systemMessage, String input, LlamaContextParams contextParams) {
    try (Arena arena = Arena.ofConfined()) {
      LlamaTemplate llamaTemplate = new LlamaTemplate(model);
      var messages = new LlamaChatMessages(
        arena,
        List.of(new LlamaChatMessage(arena, Role.SYSTEM, systemMessage), new LlamaChatMessage(arena, Role.USER, input))
      );
      return llamaTemplate.applyTemplate(arena, messages, contextParams.nCtx());
    }
  }

  private static Map<String, String> parseArgs(String[] args) {
    Map<String, String> params = new HashMap<>();
    for (int i = 0; i < args.length; i++) {
      String arg = args[i];
      if (arg.startsWith("--")) {
        String key = arg.substring(2);
        if (i + 1 < args.length && !args[i + 1].startsWith("-")) {
          params.put(key, args[++i]);
        } else {
          params.put(key, "");
        }
      } else if (arg.startsWith("-")) {
        String key = arg.substring(1);
        if (i + 1 < args.length && !args[i + 1].startsWith("-")) {
          params.put(key, args[++i]);
        } else {
          params.put(key, "");
        }
      }
    }
    return params;
  }
}
