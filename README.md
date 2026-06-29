# Llamaj.cpp

[![llamaj.cpp](https://img.shields.io/github/v/release/gravitee-io/llamaj.cpp?label=llamaj.cpp&color=orange&sort=semver)](https://github.com/gravitee-io/llamaj.cpp/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE.txt)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/gravitee-io/llamaj.cpp/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/gravitee-io/llamaj.cpp/tree/main)
[![Community Forum](https://img.shields.io/badge/Gravitee-Community%20Forum-white?logo=githubdiscussion&logoColor=white)](https://community.gravitee.io?utm_source=readme)

[![llama.cpp](https://img.shields.io/badge/llama.cpp-b9673-blue.svg)](https://github.com/ggml-org/llama.cpp/releases/tag/b9673)
[![llama.cpp license](https://img.shields.io/badge/llama.cpp%20license-MIT-green.svg)](./licenses/LICENSE-llama-cpp)

**Llamaj.cpp** is a Java and JVM port of [llama.cpp](https://github.com/ggml-org/llama.cpp) using jextract, enabling local large language model (LLM) inference through the native foreign function & memory API. It natively supports macOS M-series and Linux x86_64 with GPU acceleration; other platforms (Windows, ARM, CUDA, …) can be added through [custom builds](./docs/custom-builds/README.md).

## Requirements

- Java 25
- Maven
- macOS M-series / Linux x86_64 (other platforms via [custom builds](./docs/custom-builds/README.md))

## Installation

```xml
<dependency>
    <groupId>io.gravitee.llama.cpp</groupId>
    <artifactId>llamaj.cpp</artifactId>
    <version>1.3.0</version>
</dependency>
```

## Capabilities

Full documentation lives in **[`docs/`](./docs/README.md)** — one page per capability. New here? Start with **[Getting Started](./docs/getting-started/README.md)**.

### Core
- **[Getting Started](./docs/getting-started/README.md)** — runtime init, load a model, build a context, tokenize, and the `Arena`/`Freeable` resource model.
- **[Text Generation & Sampling](./docs/text-generation/README.md)** — stream a conversation with `ConversationState` + `DefaultLlamaIterator` and a `LlamaSampler` chain.
- **[Chat Templates](./docs/chat-templates/README.md)** — render role-tagged messages into a model-specific prompt.
- **[Log Probabilities](./docs/logprobs/README.md)** — per-token logprobs with top-N alternatives.

### Scale & serve
- **[Parallel Conversations](./docs/parallel-conversations/README.md)** — decode many conversations through one shared context with `BatchIterator`.
- **[Speculative Decoding](./docs/speculative-decoding/README.md)** — draft→verify→accept speedup (draft model or n-gram), greedy (lossless) or sampling (exact).
- **[Distributed Inference (RPC)](./docs/distributed-inference/README.md)** — offload layers/KV-cache to remote `rpc-server` backends.
- **[Quantized KV Cache](./docs/quantized-kv-cache/README.md)** — shrink KV-cache memory by quantizing the K/V cache type.

### Retrieval & multimodal
- **[Embeddings](./docs/embeddings/README.md)** — dense vectors with `LlamaEmbedder`.
- **[Reranking](./docs/reranking/README.md)** — score documents against a query with `LlamaReranker`.
- **[Multimodal (Vision & Audio)](./docs/multimodal/README.md)** — attach images and audio via `MtmdContext`.

### Advanced generation
- **[Reasoning & Tool Calls](./docs/reasoning-and-tool-calls/README.md)** — tag reasoning / answer / tool-call sections during generation.
- **[LoRA Adapters](./docs/lora-adapters/README.md)** — load and attach a GGUF LoRA adapter.

### Operations
- **[Devices & Memory](./docs/device-and-memory/README.md)** — enumerate backends/devices, query memory, read model dims and performance metrics.
- **[Logging](./docs/logging/README.md)** — custom log callback and level filtering.
- **[Custom Builds & Platform Support](./docs/custom-builds/README.md)** — build llama.cpp and regenerate the jextract bindings for other platforms.

## Quick start

```java
var arena = Arena.ofConfined();
LlamaRuntime.llama_backend_init();
LlamaRuntime.ggml_backend_load_all_from_path(arena, LlamaLibLoader.load());

var model   = new LlamaModel(arena, Path.of("models/model.gguf"), new LlamaModelParams(arena));
var context = new LlamaContext(arena, model, new LlamaContextParams(arena).nCtx(2048).nBatch(512));
var vocab   = new LlamaVocab(model);
var sampler = new LlamaSampler(arena).temperature(0.7f).topK(40).topP(0.9f, 1).seed(42);

var state = ConversationState.create(arena, context, new LlamaTokenizer(vocab, context), sampler)
    .setMaxTokens(100)
    .initialize("What is the capital of France?");

new DefaultLlamaIterator(state).stream().forEach(o -> System.out.print(o.content()));
```

See [Getting Started](./docs/getting-started/README.md) and [Text Generation & Sampling](./docs/text-generation/README.md) for the full walkthrough.

## Build from source

A platform-specific Maven profile downloads jextract + the pre-built llama.cpp native libraries, generates the FFM bindings, and installs the artifact:

```bash
# macOS (Apple Silicon)
mvn prettier:write license:format clean generate-sources -Pmacosx-aarch64 install

# Linux (x86_64) — then export the library path at runtime:
mvn prettier:write license:format clean generate-sources -Plinux-x86_64 install
export LD_LIBRARY_PATH="$HOME/.llama.cpp:$LD_LIBRARY_PATH"
```

For other platforms (Windows, ARM, CUDA, …) or your own llama.cpp build, see **[Custom Builds & Platform Support](./docs/custom-builds/README.md)**.

## Run the bundled CLI

```bash
java -jar llamaj.cpp-<version>.jar --model models/model.gguf \
  --system 'You are a helpful assistant.'
```

## License & attribution

- llamaj.cpp is licensed under the [Apache License 2.0](./LICENSE.txt).
- It binds to the native **llama.cpp / ggml** libraries, which are [MIT-licensed](./licenses/LICENSE-llama-cpp).

## Community

Questions and discussion: [Gravitee Community Forum](https://community.gravitee.io?utm_source=readme).
