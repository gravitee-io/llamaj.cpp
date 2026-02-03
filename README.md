#  Llamaj.cpp

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/gravitee-io/llamaj.cpp/LICENSE.txt)
[![Releases](https://img.shields.io/badge/semantic--release-conventional%20commits-e10079?logo=semantic-release)](https://github.com/gravitee-io/llamaj.cpp/releases)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/gravitee-io/llamaj.cpp/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/gravitee-io/llamaj.cpp/tree/main)
[![Community Forum](https://img.shields.io/badge/Gravitee-Community%20Forum-white?logo=githubdiscussion&logoColor=white)](https://community.gravitee.io?utm_source=readme)

**Llamaj.cpp** is a Java and JVM port of llama.cpp using jextract, enabling local large language model (LLM) inference through native foreign function & memory API interop. Natively supports macOS M-series and Linux x86_64 with GPU acceleration. Platform and hardware support (Windows, ARM, CUDA, etc.) can be extended through custom builds.

## Keywords

`llama.cpp` · `java` · `jvm` · `llm` · `large language models` · `inference` · `ai` · `native interop` · `foreign function & memory api` · `jextract`

## Requirements

- Java 25
- mvn
- MacOS M-series / Linux x86_64 (CPU) (you can check the last section if you do not see your platform here)

## How to use

Include the dependency in your pom.xml
```
    <dependencies>
        ...
        <dependency>
            <groupId>io.gravitee.llama.cpp</groupId>
            <artifactId>llamaj.cpp</artifactId>
            <version>x.x.x</version>
        </dependency>
    </dependencies>
```

> **Note:** All examples below use `LlamaVocab` to handle tokenization. It's obtained from a loaded `LlamaModel` and is essential for converting between tokens and text representations.

### Example 1: Basic Conversation

```java
import io.gravitee.llama.cpp.*;
import java.lang.foreign.Arena;
import java.nio.file.Path;

public class BasicExample {
    public static void main(String[] args) {
        var arena = Arena.ofConfined();

        // Initialize runtime
        LlamaRuntime.llama_backend_init();

        // Load model
        var modelParams = new LlamaModelParams(arena);
        var model = new LlamaModel(arena, Path.of("models/model.gguf"), modelParams);

        // Create context
        var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512);
        var context = new LlamaContext(model, contextParams);

        // Set up tokenizer and sampler
        var vocab = new LlamaVocab(model);
        var tokenizer = new LlamaTokenizer(vocab, context);
        var sampler = new LlamaSampler(arena)
            .temperature(0.7f)
            .topK(40)
            .topP(0.9f, 1)
            .seed(42);

        // Create conversation state
        var state = ConversationState.create(arena, context, tokenizer, sampler, 0)
            .setMaxTokens(100)
            .initialize("What is the capital of France?");

        // Generate response
        var iterator = new DefaultLlamaIterator(state);
        while (iterator.hasNext()) {
            var output = iterator.next();
            System.out.print(output.text());
        }

        // Cleanup
        context.free();
        sampler.free();
        model.free();
        LlamaRuntime.llama_backend_free();
    }
}
```

### Example 2: Log Probabilities

Enable log-probability collection to inspect the model's confidence at each token position.
Set `topLogprobs` to the number of top-alternative tokens you want alongside the sampled one (0 = disabled, no overhead):

```java
import io.gravitee.llama.cpp.*;
import java.lang.foreign.Arena;
import java.nio.file.Path;

public class LogprobsExample {
    public static void main(String[] args) {
        var arena = Arena.ofConfined();
        LlamaRuntime.llama_backend_init();

        var model = new LlamaModel(arena, Path.of("models/model.gguf"), new LlamaModelParams(arena));
        var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512);
        var context = new LlamaContext(arena, model, contextParams);
        var vocab = new LlamaVocab(model);
        var tokenizer = new LlamaTokenizer(vocab, context);
        var sampler = new LlamaSampler(arena).temperature(0.7f).seed(42);

        var state = ConversationState.create(arena, context, tokenizer, sampler)
            .setMaxTokens(50)
            .setTopLogprobs(5)   // return top-5 alternatives at every token position
            .initialize("What is the capital of France?");

        var iterator = new DefaultLlamaIterator(state);
        while (iterator.hasNext()) {
            var output = iterator.next();
            System.out.print(output.text());

            Logprobs lp = output.logprobs();
            System.out.printf("%n  chosen: \"%s\"  logprob=%.4f%n",
                lp.chosenToken().token(), lp.chosenToken().logprob());
            lp.topLogprobs().forEach(t ->
                System.out.printf("    alt: \"%s\"  logprob=%.4f%n", t.token(), t.logprob()));
        }

        context.free();
        sampler.free();
        model.free();
        LlamaRuntime.llama_backend_free();
    }
}
```

Each `LlamaOutput` carries a `Logprobs` object with:
- `chosenToken()` — the token that was sampled, its text, vocabulary ID, log-probability, and raw UTF-8 bytes
- `topLogprobs()` — up to N alternatives sorted by descending log-probability; the chosen token is always included

When `topLogprobs` is `0` (the default), `output.logprobs()` is `null` and no logit processing is done.

### Example 3: Parallel Conversations

Process multiple conversations simultaneously in a single batch:

```java
import io.gravitee.llama.cpp.*;

import java.lang.foreign.Arena;
import java.nio.file.Path;

public class ParallelExample {
    public static void main(String[] args) {
        var arena = Arena.ofConfined();

        // Initialize runtime
        LlamaRuntime.llama_backend_init();

        // Load model
        var modelParams = new LlamaModelParams(arena);
        var model = new LlamaModel(arena, Path.of("models/model.gguf"), modelParams);

        // Create context with multi-sequence support
        var contextParams = new LlamaContextParams(arena)
                .nCtx(2048)
                .nBatch(512)
                .nSeqMax(4);  // Support up to 4 parallel conversations
        var context = new LlamaContext(model, contextParams);

        // Set up shared tokenizer and sampler
        var vocab = new LlamaVocab(model);
        var tokenizer = new LlamaTokenizer(vocab, context);
        var sampler = new LlamaSampler(arena).temperature(0.7f).seed(42);

        // Create multiple conversation states with unique sequence IDs
        var state1 = ConversationState.create(arena, context, tokenizer, sampler, 0)
                .setMaxTokens(100).initialize("What is the capital of France?");
        var state2 = ConversationState.create(arena, context, tokenizer, sampler, 1)
                .setMaxTokens(100).initialize("What is the capital of England?");
        var state3 = ConversationState.create(arena, context, tokenizer, sampler, 2)
                .setMaxTokens(100).initialize("What is the capital of Poland?");

        // Create parallel iterator - prompts are auto-processed when states are added
        var parallel = new BatchIterator(arena, context, 512, 4)
                .addState(state1)
                .addState(state2)
                .addState(state3);

        // Generate tokens in parallel
        System.out.println("=== Parallel Generation ===");
        while (parallel.hasNext()) {
            // Each hasNext() generates tokens for all active conversations
            // Get all outputs from this batch (one per active conversation)
            var outputs = parallel.getOutputs();
            for (var output : outputs) {
                System.out.println("Seq " + output.sequenceId() + ": " + output.text());
            }
        }
        System.out.println();

        // Print results
        System.out.println("Conversation 1: " + state1.getAnswer());
        System.out.println("  Tokens: " + state1.getAnswerTokens());
        System.out.println("Conversation 2: " + state2.getAnswer());
        System.out.println("  Tokens: " + state2.getAnswerTokens());
        System.out.println("Conversation 3: " + state3.getAnswer());
        System.out.println("  Tokens: " + state3.getAnswerTokens());

        // Cleanup
        parallel.free();
        context.free();
        sampler.free();
        model.free();
        LlamaRuntime.llama_backend_free();
    }
}
```

### Example 4: Distributed Inference with RPC

Offload model weights and KV-cache to remote machines using the RPC backend.
When using `--rpc`, weights are loaded **exclusively** on the remote servers -- the local GPU is not used.

Start RPC server nodes first (see [containers/README.md](containers/README.md)):

```bash
# On the remote machine (or another terminal)
./scripts/start-rpc-server.sh
```

Then connect from Java:

```java
import io.gravitee.llama.cpp.*;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;

public class RpcExample {
    public static void main(String[] args) {
        var arena = Arena.ofConfined();

        // Initialize runtime
        String libPath = LlamaLibLoader.load();
        LlamaRuntime.llama_backend_init();

        // Register remote RPC servers -- returns their device handles
        var rpcDevices = BackendRegistry.addRpcServer(arena, "127.0.0.1:50052");

        // Print all discovered backends and devices
        BackendRegistry.printSummary();

        // Load model, restricting offloading to only the RPC devices
        var modelParams = new LlamaModelParams(arena)
            .devices(arena, rpcDevices)
            .nGpuLayers(999);
        var model = new LlamaModel(arena, Path.of("models/model.gguf"), modelParams);

        // Everything else works exactly the same as local inference
        var contextParams = new LlamaContextParams(arena).nCtx(2048).nBatch(512);
        var context = new LlamaContext(model, contextParams);
        var vocab = new LlamaVocab(model);
        var tokenizer = new LlamaTokenizer(vocab, context);
        var sampler = new LlamaSampler(arena).temperature(0.7f).seed(42);

        var state = ConversationState.create(arena, context, tokenizer, sampler, 0)
            .setMaxTokens(100)
            .initialize("What is the capital of France?");

        var iterator = new DefaultLlamaIterator(state);
        while (iterator.hasNext()) {
            System.out.print(iterator.next().text());
        }

        context.free();
        sampler.free();
        model.free();
        LlamaRuntime.llama_backend_free();
    }
}
```

Or from the CLI:

```bash
$ java --enable-preview --enable-native-access=ALL-UNNAMED \
  -jar llamaj.cpp-<version>.jar \
  --model models/model.gguf \
  --rpc 127.0.0.1:50052
```

Multiple RPC servers:

```bash
$ java --enable-preview --enable-native-access=ALL-UNNAMED \
  -jar llamaj.cpp-<version>.jar \
  --model models/model.gguf \
  --rpc 192.168.1.10:50052,192.168.1.11:50052
```

## Build

The build uses a **platform-specific Maven profile** to download the correct jextract tool and pre-built llama.cpp native libraries, generate the Java FFM bindings, format the code, apply license headers, and install the artifact to your local Maven repository.

**macOS (Apple Silicon):**

```bash
cd llamaj.cpp/
mvn prettier:write license:format clean generate-sources -Pmacosx-aarch64 install
```

**Linux (x86_64):**

```bash
cd llamaj.cpp/
mvn prettier:write license:format clean generate-sources -Plinux-x86_64 install
```

> On Linux, you also need to set the library path at runtime:
> ```bash
> export LD_LIBRARY_PATH="$HOME/.llama.cpp:$LD_LIBRARY_PATH"
> ```

## Run

```bash
$ mvn exec:java -Dexec.mainClass=io.gravitee.llama.cpp.Main \
    -Dexec.args="--model /path/to/model/model.gguf --system 'You are a helpful assistant. Answer question to the best of your ability'"
```

or

```bash
$ java --enable-preview -jar llamaj.cpp-<version>.jar \
  --model models/model.gguf \
  --system 'You are a helpful assistant. Answer question to the best of your ability'
```

On linux, don't forget to link your libraries with the environment variable below:
```bash
$ export LD_LIBRARY_PATH="$HOME/.llama.cpp:$LD_LIBRARY_PATH"
```

There are plenty of models on HuggingFace, we suggest the one [here](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)

### Usage
```
Usage: java -jar llamaj.cpp-<version>.jar --model <path_to_gguf_model> [options...]
Options:
--system <message>       : System message (default: "You are a helpful AI assistant.")
--n_gpu_layers <int>     : Number of GPU layers (default: 999)
--use_mlock <boolean>    : Use mlock (default: true)
--use_mmap <boolean>     : Use mmap (default: true)
--rpc <endpoints>        : Comma-separated RPC server endpoints for distributed inference
                           (e.g., "127.0.0.1:50052,192.168.1.11:50052")
                           When set, weights are offloaded exclusively to the remote servers
--temperature <float>    : Sampler temperature (default: 0.4)
--min_p <float>          : Sampler min_p (default: 0.1)
--min_p_window <int>     : Sampler min_p_window (default: 40)
--top_k <int>            : Sampler top_k (default: 10)
--top_p <float>          : Sampler top_p (default: 0.2)
--top_p_window <int>     : Sampler top_p_window (default: 10)
--seed <long>            : Sampler seed (default: random)
--n_ctx <int>            : Context size (default: 512)
--n_batch <int>          : Batch size (default: 512)
--n_seq_max <int>        : Max sequence length (default: 512)
--quota <int>            : Iterator quota (default: 512)
--n_keep <int>         : Tokens to keep when exceeding ctx size (default: 256)
--log_level <level>      : Logging level (ERROR, WARN, INFO, DEBUG, default: ERROR)
```

## Use your own llama.cpp build

1. Clone `llama.cpp` repository

> Make sure the jextract folder is in the same path level as your repository

```bash
$ git clone https://github.com/ggml-org/llama.cpp
$ cd llama.cpp
```

2. Compile sources

> Make sure you have gcc / g++ compiler

```bash
$ gcc --help
$ g++ --help
```

On Linux:
```bash
$ cmake -B build
$ cmake --build build --config Release -j $(nproc)  
```

On MacOs:
```bash
$ cmake -B build
$ cmake --build build --config Release  -j $(sysctl -n hw.ncpu)
```

If you wish to build llama.cpp with particular configuration (CUDA, OpenBLAS, AVX2, AVX512, ...)
Please refer to the [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) documentation

3. Link sources

You can use the environment variable `LLAMA_CPP_LIB_PATH=/path/to/llama.cpp/build/bin/`
This will directly load the dynamically shared object library files (`.so` for linux, `.dylib` for macos) 
You can also decide to copy these files into a temporary folder using the environment variable `LLAMA_CPP_USE_TMP_LIB_PATH=true`
The path temporary file will be used to load the shared object libraries

## Beyond Apple M-Series and Linux x86_64

To add support for other platforms (Windows, ARM, CUDA, etc.), follow this approach:

### 1. Build llama.cpp

Clone and build llama.cpp for your target platform:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

### 2. Generate FFM API Bindings with jextract

Download jextract for your platform from [OpenJDK early-access builds](https://download.java.net/java/early_access/jextract/25/2/), then generate the Java bindings:

```bash
# Example for Windows x86_64
jextract -t io.gravitee.llama.cpp.windows.x86_64 \
  --include-dir /path/to/llama.cpp/ggml/include \
  --include-dir /path/to/llama.cpp/include \
  --output src/main/java \
  --header-class-name llama_h \
  /path/to/llama.cpp/tools/mtmd/mtmd.h \
  /path/to/llama.cpp/tools/mtmd/mtmd-helper.h \
  /path/to/llama.cpp/include/llama.h \
  /path/to/llama.cpp/ggml/include/ggml-rpc.h
```

### 3. Post-process Generated Sources

Check the generated sources and apply any necessary fixes (e.g., visibility modifiers, fully-qualified method calls).

### 4. Build the Bindings JAR

Compile the generated sources and build a JAR using your own build system (Maven, Gradle, etc.).

### 5. Integrate into Your Classpath

Add the generated JAR to your project's classpath and ensure the native libraries from step 1 are available at runtime.