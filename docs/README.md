# llamaj.cpp — Capabilities

One folder per capability, one page each. Start with **[Getting Started](./getting-started/README.md)** — every other page builds on it.

### Core
| Capability | What it does |
| --- | --- |
| [Getting Started](./getting-started/README.md) | Initialize the runtime, load a GGUF model, build a context, and tokenize — plus the `Arena`/`Freeable` resource model. |
| [Text Generation & Sampling](./text-generation/README.md) | Stream a conversation token-by-token via `ConversationState` + `DefaultLlamaIterator`, controlled by a `LlamaSampler` chain, max tokens, and stop strings. |
| [Chat Templates](./chat-templates/README.md) | Render role-tagged messages into a model-specific prompt with `LlamaTemplate` and the `Role` enum. |
| [Log Probabilities](./logprobs/README.md) | Collect per-token logprobs (sampled token + top-N alternatives) via `setTopLogprobs`. |

### Scale & serve
| Capability | What it does |
| --- | --- |
| [Parallel Conversations](./parallel-conversations/README.md) | Decode many conversations through one shared context with `BatchIterator`, isolated by sequence id. |
| [Speculative Decoding](./speculative-decoding/README.md) | Draft→verify→accept speedup via a draft model or n-gram prompt-lookup; greedy (lossless) or sampling (exact). |
| [Distributed Inference (RPC)](./distributed-inference/README.md) | Offload layers and KV-cache to remote `rpc-server` backends across machines. |
| [Quantized KV Cache](./quantized-kv-cache/README.md) | Shrink KV-cache memory by quantizing the K/V cache `ggml_type` (flash attention for a quantized V cache). |

### Retrieval & multimodal
| Capability | What it does |
| --- | --- |
| [Embeddings](./embeddings/README.md) | Turn text into dense vectors with `LlamaEmbedder` over an embedding-mode context. |
| [Reranking](./reranking/README.md) | Score documents against a query with `LlamaReranker` (BERT cross-encoders and Qwen3-style rerankers). |
| [Multimodal (Vision & Audio)](./multimodal/README.md) | Attach images and audio to a prompt via `MtmdContext` (mmproj/clip projector). |

### Advanced generation
| Capability | What it does |
| --- | --- |
| [Reasoning & Tool Calls](./reasoning-and-tool-calls/README.md) | Tag reasoning / answer / tool-call sections during generation for per-section token counts and `TOOL_CALL` detection. |
| [LoRA Adapters](./lora-adapters/README.md) | Load a GGUF LoRA adapter and attach it to a model with `initLoraAdapter`. |

### Operations
| Capability | What it does |
| --- | --- |
| [Devices & Memory](./device-and-memory/README.md) | Enumerate backends/devices, query CPU/GPU/RPC memory, read model dims, and read performance metrics. |
| [Logging](./logging/README.md) | Install a custom log callback over llama.cpp/ggml and filter by level. |
| [Custom Builds & Platform Support](./custom-builds/README.md) | Build llama.cpp and regenerate the jextract FFM bindings for other platforms (Windows, ARM, CUDA, …). |
