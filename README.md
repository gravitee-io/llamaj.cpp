#  Llamaj.cpp

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/gravitee-io/llamaj.cpp/LICENSE.txt)
[![Releases](https://img.shields.io/badge/semantic--release-conventional%20commits-e10079?logo=semantic-release)](https://github.com/gravitee-io/llamaj.cpp/releases)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/gravitee-io/llamaj.cpp/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/gravitee-io/llamaj.cpp/tree/main)
[![Community Forum](https://img.shields.io/badge/Gravitee-Community%20Forum-white?logo=githubdiscussion&logoColor=white)](https://community.gravitee.io?utm_source=readme)

llamaj.cpp (contraction of llama.cpp and java/jextract) is a port of llama.cpp in the JVM using jextract.

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
            <artifactId>llamaj-cpp</artifactId>
            <version>x.x.x</version>
        <dependency>
    </dependencies>
```

### Example 1: Basic Conversation

```java
import io.gravitee.llama.cpp.*;
import java.lang.foreign.Arena;

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

### Example 2: Parallel Conversations

Process multiple conversations simultaneously in a single batch:

```java
import io.gravitee.llama.cpp.*;

import java.lang.foreign.Arena;

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

## Build

1. Get `jextract`

> Make sure the `jextract` folder is in the same path level as your repository

On Linux:

Since we are using JDK 25 you can download a prebuilt version of `jextract`

```bash
$ wget https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_linux-x64_bin.tar.gz
$ tar -xzf openjdk-25-jextract+2-4_linux-x64_bin.tar.gz
$ rm openjdk-25-jextract+2-4_linux-x64_bin.tar.gz
$ echo 'export PATH="$(pwd)/jextract/bin:$PATH"' >> ~/.bashrc
```

On macOS:

Since we are using JDK 25 you can download a prebuilt version of `jextract`

```bash
$ wget https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_macos-aarch64_bin.tar.gz
$ tar -xzf openjdk-25-jextract+2-4_macos-aarch64_bin.tar.gz
$ rm openjdk-25-jextract+2-4_macos-aarch64_bin.tar.gz
$ echo 'export PATH="$(pwd)/jextract/bin:$PATH"' >> ~/.zshrc
```

> Note: On macOS Catalina or later, you may need to remove the quarantine attribute from the jextract binaries:
> ```bash
> sudo xattr -r -d com.apple.quarantine jextract
> ```

3. Clone llama.cpp

> Make sure `llama.cpp` folder is in the same path level as your repository

```bash
$ git clone https://github.com/ggml-org/llama.cpp
```

5. Download binaries and generate the sources

```bash
$ mkdir $HOME/.llama.cpp
$ cd llamaj.cpp/
$ mvn clean generate-sources -Pmacosx-aarch64,linux-x86_64
$ export LLAMA_CPP_LIB_PATH="$HOME_DIR/llamaj.cpp/target/generated-sources/<<macosx|linux>>/<<x86_64|aarch64>>"
$ mvn install
```

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

While we don't support other platforms/architecture pair out-of-the-box for many reasons, you can still manage to use 
gravitee-io/llamaj.cpp:

1. Build llama.cpp on your infrastructure
2. Add the *.so or *.dylib to ~/.llama.cpp/ or use the `LLAMA_CPP_LIB_PATH` and `LD_LIBRARY_PATH`
3. Build the according java bindings using `jextract` (without `--source` option) and bundle them into a jar
```bash
$ jextract -t io.gravitee.llama.cpp.<os>.<platform>\
    --include-dir ggml/include \
    --output /path/to/your/output include/llama.h
$ jar cf <name-of-your-file>.jar -C . .
```
- Put the `jextract` source in `io.gravitee.llama.cpp.<os>.<arch>`:
    - `io.gravitee.llama.cpp.macosx.x86_64`
    - `io.gravitee.llama.cpp.linux.aarch64`
    - `io.gravitee.llama.cpp.windows.x86_64`
    - `io.gravitee.llama.cpp.windows.aarch64`

4. Add it to your classpath:

gravitee-io/llamaj.cpp will pick up at runtime the os and architecture and will call the according bindings using reflection.
