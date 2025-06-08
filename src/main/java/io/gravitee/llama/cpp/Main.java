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

import static io.gravitee.llama.cpp.LlamaRuntime.llama_backend_init;
import static java.lang.Boolean.parseBoolean;
import static java.lang.Float.parseFloat;
import static java.lang.Integer.parseInt;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.*;

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
    int nSeqMax = parseInt(params.getOrDefault("n_seq_max", "64"));
    int quota = parseInt(params.getOrDefault("quota", "512"));
    int nKeep = parseInt(params.getOrDefault("nKeep", "256"));

    String logLevelStr = params.getOrDefault("log_level", "ERROR").toUpperCase();
    LlamaLogLevel logLevel;
    try {
      logLevel = LlamaLogLevel.valueOf(logLevelStr);
    } catch (IllegalArgumentException e) {
      System.err.println("Invalid log level: %s. Using default: ERROR.".formatted(logLevelStr));
      logLevel = LlamaLogLevel.ERROR;
    }

    if (modelGguf == null) {
      System.err.println("Usage: java -jar llamaj.cpp-<version>.jar  --model <path_to_gguf_model> [options...]");
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
      System.err.println("  --n_keep <int>         : Tokens to keep when exceeding ctx size (default: 256)");
      System.err.println("  --log_level <level>      : Logging level (ERROR, WARN, INFO, DEBUG, default: ERROR)");
      System.exit(1);
    }

    LlamaLibLoader.load();
    llama_backend_init();

    System.out.println("****************************");
    System.out.println("Number of devices registered: " + LlamaRuntime.ggml_backend_reg_count());
    System.out.println("****************************");

    var logger = new LlamaLogger(ARENA);
    logger.setLogging(logLevel); // Applying the parsed log level

    var modelParameters = new LlamaModelParams(ARENA);
    modelParameters.nGpuLayers(nGpuLayers).useMlock(useMlock).useMmap(useMmap);

    var model = new LlamaModel(ARENA, Path.of(modelGguf).toAbsolutePath(), modelParameters);
    var vocab = new LlamaVocab(model);

    var contextParams = new LlamaContextParams(ARENA).nCtx(nCtx).nBatch(nBatch).nSeqMax(nSeqMax);

    LlamaSampler sampler = new LlamaSampler(ARENA)
      .temperature(temperature)
      .minP(minP, minPWindow)
      .topK(topK)
      .topP(topP, topPWindow)
      .seed(seed);

    List<LlamaChatMessage> messages = new ArrayList<>();
    messages.add(new LlamaChatMessage(ARENA, Role.SYSTEM, systemMessage));

    String input = "";

    var context = new LlamaContext(model, contextParams);
    var tokenizer = new LlamaTokenizer(vocab, context);

    var messageTrimmer = new MessageTrimmer(tokenizer, context.nCtx(), nKeep, systemMessage);

    while (!input.trim().equals("bye")) {
      Scanner scanIn = new Scanner(System.in);
      System.out.print("Please enter your prompt: ");
      input = scanIn.nextLine();

      if (input.isBlank()) {
        break;
      }

      messages.add(new LlamaChatMessage(ARENA, Role.USER, input));

      String prompt = buildPrompt(model, messageTrimmer.trimMessages(messages), contextParams);

      LlamaIterator iterator = new SimpleLlamaIterator(ARENA, context, tokenizer, sampler)
        .setQuota(quota)
        .initialize(prompt);

      var answer = iterator.stream().map(LlamaOutput::content).peek(System.out::print).reduce((a, b) -> a + b).orElse("");

      messages.add(new LlamaChatMessage(ARENA, Role.ASSISTANT, answer));
      messages = messageTrimmer.trimMessages(messages);

      context.kvCacheClear();

      System.out.println();
    }

    context.free();
    sampler.free();
    model.free();

    llama_backend_init();
  }

  private static String buildPrompt(LlamaModel model, List<LlamaChatMessage> messages, LlamaContextParams contextParams) {
    try (Arena arena = Arena.ofConfined()) {
      return new LlamaTemplate(model).applyTemplate(arena, new LlamaChatMessages(arena, messages), contextParams.nCtx());
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

  private record MessageTrimmer(LlamaTokenizer tokenizer, int contextSize, int nKeep, String systemMessage) {
    public List<LlamaChatMessage> trimMessages(List<LlamaChatMessage> fullHistory) {
      try (Arena arena = Arena.ofConfined()) {
        List<LlamaChatMessage> trimmed = new ArrayList<>();
        int totalTokens = tokenizer.tokenize(arena, systemMessage).size();

        LlamaChatMessage firstUserMessage = fullHistory
          .stream()
          .filter(m -> m.getRole() == Role.USER)
          .findFirst()
          .orElse(null);

        int firstUserTokens = 0;
        if (firstUserMessage != null) {
          firstUserTokens = tokenizer.tokenize(arena, firstUserMessage.getContent().strip()).size();
        }

        ListIterator<LlamaChatMessage> it = fullHistory.listIterator(fullHistory.size());
        int added = 0;

        while (it.hasPrevious()) {
          LlamaChatMessage msg = it.previous();

          if (msg.getRole() == Role.SYSTEM) continue;
          if (msg == firstUserMessage) continue;

          int tokenCount = tokenizer.tokenize(arena, msg.getContent().strip()).size();

          if ((totalTokens + tokenCount) > (contextSize - firstUserTokens)) break;

          trimmed.addFirst(msg);
          totalTokens += tokenCount;

          if (++added >= nKeep) break;
        }

        trimmed.addFirst(firstUserMessage);
        trimmed.addFirst(new LlamaChatMessage(Main.ARENA, Role.SYSTEM, systemMessage));
        return trimmed;
      }
    }
  }
}
