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

import static io.gravitee.llama.cpp.LlamaRuntime.*;
import static java.lang.Boolean.parseBoolean;
import static java.lang.Integer.parseInt;
import static java.util.Optional.ofNullable;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class Main {

  public static final Arena ARENA = Arena.ofAuto();

  public enum SamplingStrategy {
    DETERMINISTIC,
    CLASSIC_CHAT,
    FOCUSED,
    BALANCED,
    ADAPTIVE,
    CONSTRAINED;

    public static SamplingStrategy fromString(String strategy) {
      if (strategy == null || strategy.isBlank()) {
        return CLASSIC_CHAT;
      }
      return valueOf(strategy.toUpperCase());
    }
  }

  public static void main(String[] args) {
    Map<String, String> params = parseArgs(args);

    String modelGguf = params.get("model");
    String lora = params.get("lora");
    String systemMessage = params.getOrDefault("system", "You are a helpful AI assistant.");

    int nGpuLayers = parseInt(params.getOrDefault("n_gpu_layers", "999"));
    boolean useMlock = parseBoolean(params.getOrDefault("use_mlock", "true"));
    boolean useMmap = parseBoolean(params.getOrDefault("use_mmap", "true"));

    int nCtx = parseInt(params.getOrDefault("n_ctx", "4096"));
    int nBatch = parseInt(params.getOrDefault("n_batch", "4096"));

    int quota = parseInt(params.getOrDefault("quota", String.valueOf(nCtx)));
    int nKeep = parseInt(params.getOrDefault("nKeep", String.valueOf(nCtx)));

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
      System.exit(1);
    }

    var libPath = LlamaLibLoader.load();
    llama_backend_init();
    ggml_backend_load_all_from_path(ARENA, libPath);

    System.out.println("****************************");
    System.out.println("Libraries loaded at: " + libPath);
    System.out.println("Number of devices registered: " + ggml_backend_reg_count());
    System.out.println("****************************");

    var logger = new LlamaLogger(ARENA);
    logger.setLogging(logLevel);

    var modelParameters = new LlamaModelParams(ARENA);
    modelParameters.nGpuLayers(nGpuLayers).useMlock(useMlock).useMmap(useMmap);

    var model = new LlamaModel(ARENA, Path.of(modelGguf).toAbsolutePath(), modelParameters);

    ofNullable(lora).ifPresent(path -> model.initLoraAdapter(ARENA, Path.of(path).toAbsolutePath()));

    var vocab = new LlamaVocab(model);
    var contextParams = new LlamaContextParams(ARENA).nCtx(nCtx).nBatch(nBatch);

    SamplingStrategy strategy = SamplingStrategy.fromString(params.get("strategy"));
    LlamaContext context = new LlamaContext(model, contextParams);
    LlamaSampler sampler = llamaSampler(strategy, context, vocab, params);

    List<LlamaChatMessage> messages = new ArrayList<>();
    messages.add(new LlamaChatMessage(ARENA, Role.SYSTEM, systemMessage));

    String input = "";
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

      LlamaIterator iterator = new DefaultLlamaIterator(ARENA, context, tokenizer, sampler)
        .setMaxTokens(quota)
        .initialize(prompt);

      var answer = iterator.stream().map(LlamaOutput::content).peek(System.out::print).reduce((a, b) -> a + b).orElse("");

      if (LlamaLogLevel.DEBUG.equals(logLevel)) {
        System.out.println("Input tokens: " + iterator.getInputTokens());
        System.out.println("Output tokens: " + iterator.getOutputTokens());
        System.out.println("Finish Reason: " + iterator.getFinishReason());
      }
      messages.add(new LlamaChatMessage(ARENA, Role.ASSISTANT, answer));
      messages = messageTrimmer.trimMessages(messages);

      context.clearCache();

      System.out.println();
    }

    context.free();
    sampler.free();
    model.free();

    llama_backend_free();
    ggml_backend_free();
  }

  public static LlamaSampler llamaSampler(
    SamplingStrategy strategy,
    LlamaContext context,
    LlamaVocab vocab,
    Map<String, String> params
  ) {
    float temperature = Float.parseFloat(params.getOrDefault("temperature", "0.7"));
    int topK = Integer.parseInt(params.getOrDefault("top_k", "40"));
    float topP = Float.parseFloat(params.getOrDefault("top_p", "0.9"));
    int topPWindow = Integer.parseInt(params.getOrDefault("top_p_window", "1"));
    float minP = Float.parseFloat(params.getOrDefault("min_p", "0.1"));
    int minPWindow = Integer.parseInt(params.getOrDefault("min_p_window", "1"));
    int seed = Integer.parseInt(params.getOrDefault("seed", "42"));

    int penaltyLastN = Integer.parseInt(params.getOrDefault("penalty_last_n", String.valueOf(context.nCtx())));
    float penaltyRepeat = Float.parseFloat(params.getOrDefault("penalty_repeat", "1.5"));
    float penaltyFreq = Float.parseFloat(params.getOrDefault("penalty_freq", "0.1"));
    float penaltyPresent = Float.parseFloat(params.getOrDefault("penalty_present", "0.1"));

    float mirostatTau = Float.parseFloat(params.getOrDefault("mirostat_tau", "5.0"));
    float mirostatEta = Float.parseFloat(params.getOrDefault("mirostat_eta", "0.1"));

    return switch (strategy) {
      case DETERMINISTIC -> new LlamaSampler(ARENA).greedy().seed(seed);
      case CLASSIC_CHAT -> new LlamaSampler(ARENA)
        .temperature(temperature)
        .topP(topP, topPWindow)
        .penalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent)
        .seed(seed);
      case FOCUSED -> new LlamaSampler(ARENA)
        .temperature(temperature)
        .topK(topK)
        .penalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent)
        .seed(seed);
      case BALANCED -> new LlamaSampler(ARENA)
        .temperature(temperature)
        .minP(minP, minPWindow)
        .penalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent)
        .seed(seed);
      case ADAPTIVE -> new LlamaSampler(ARENA)
        .mirostat(seed, mirostatTau, mirostatEta)
        .penalties(context.nCtx(), penaltyRepeat, penaltyFreq, penaltyPresent)
        .seed(seed);
      case CONSTRAINED -> new LlamaSampler(ARENA)
        .topP(topP, topPWindow)
        .grammar(vocab, safeRead(params.get("grammar")), params.getOrDefault("grammar_root", "root"))
        .seed(seed);
    };
  }

  private static String safeRead(String grammar) {
    try {
      return String.join("\n", Files.readAllLines(Path.of(grammar)));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
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

        ListIterator<LlamaChatMessage> it = fullHistory.listIterator(fullHistory.size());
        int added = 0;

        while (it.hasPrevious()) {
          LlamaChatMessage msg = it.previous();

          if (msg.getRole() == Role.SYSTEM) continue;

          int tokenCount = tokenizer.tokenize(arena, msg.getContent().strip()).size();

          if ((totalTokens + tokenCount) > contextSize) break;

          trimmed.addFirst(msg);
          totalTokens += tokenCount;

          if (++added >= nKeep) break;
        }

        trimmed.addFirst(new LlamaChatMessage(Main.ARENA, Role.SYSTEM, systemMessage));
        return trimmed;
      }
    }
  }
}
