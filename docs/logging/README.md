# Logging

> Install a custom log callback over llama.cpp / ggml and filter messages by severity level.

## Overview
llama.cpp and its ggml backends emit diagnostic messages (model loading, backend
selection, performance, warnings). `LlamaLogger` registers a native log callback
(`llama_log_set` / `ggml_log_callback`) that routes those messages to a Java
`Consumer<String>`, with a `LlamaLogLevel` threshold to drop messages below the
level you care about. Use it to silence noisy startup logs, raise the threshold to
warnings/errors in production, or pipe llama.cpp output into your own logging
framework (SLF4J, Logback, etc.).

## Key types
- `LlamaLogger` — `ArenaAware` handler that installs the native log callback; created with an `Arena`.
- `LlamaLogLevel` — severity enum used as the filter threshold (`NONE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `CONT`).

## Usage
```java
import io.gravitee.llama.cpp.*;
import java.lang.foreign.Arena;

Arena arena = Arena.ofConfined();

// The logger must be installed before loading models/contexts so you capture
// backend and model-load messages.
var logger = new LlamaLogger(arena);

// 1) Simplest form: threshold only — messages are written to System.out.
logger.setLogging(LlamaLogLevel.DEBUG);

// 2) Custom sink: forward llama.cpp / ggml output to your own logger.
logger.setLogging(LlamaLogLevel.WARN, message -> myLogger.warn(message));

// From here on, model and context creation emit messages through the callback.
var modelParameters = new LlamaModelParams(arena);
var model = new LlamaModel(arena, modelPath, modelParameters);
var context = new LlamaContext(arena, model, new LlamaContextParams(arena));
```

## Options

### `LlamaLogLevel` (threshold, lowest to highest)
| Value   | Effect when used as threshold                                  |
|---------|----------------------------------------------------------------|
| `NONE`  | Logs everything (no message is filtered out).                  |
| `DEBUG` | Logs `DEBUG` and above — verbose, includes per-token detail.   |
| `INFO`  | Logs `INFO`, `WARN`, `ERROR`, `CONT`.                          |
| `WARN`  | Logs warnings and errors only.                                 |
| `ERROR` | Logs errors (and `CONT` continuation lines) only.             |
| `CONT`  | Highest ordinal; continuation marker for multi-part messages.  |

### `setLogging` overloads
| Method                                              | Behavior                                              |
|-----------------------------------------------------|-------------------------------------------------------|
| `setLogging(LlamaLogLevel level)`                   | Threshold only; sink defaults to `System.out::print`. |
| `setLogging(LlamaLogLevel level, Consumer<String>)` | Threshold plus a custom message sink.                 |

## Notes
- Filtering rule: a message is delivered when its native level is `>=` the
  configured threshold's `ordinal()`. So `DEBUG` is the most verbose useful
  threshold, while `NONE` (ordinal 0) filters nothing.
- The enum ordinals map directly onto ggml's log levels — keep the declaration
  order (`NONE, DEBUG, INFO, WARN, ERROR, CONT`) intact.
- `LlamaLogger` is `ArenaAware`: the native callback stub is allocated from the
  `Arena` you pass in. Keep that `Arena` (and the `LlamaLogger`) alive for as long
  as you want logging active; closing/freeing the `Arena` invalidates the callback.
- Install the logger early — before `new LlamaModel(...)` / `new LlamaContext(...)`
  — to capture backend registration and model-load diagnostics.
- `llama_log_set` is process-global: the last installed callback wins for the
  whole JVM, not per model or per context.
- The sink runs on llama.cpp's calling (native) thread; keep it cheap and
  thread-safe, and avoid throwing out of the `Consumer`.

## See also
- [Getting Started](../getting-started/README.md) — backend init and the load order this logger should precede.
- [Devices & Memory](../device-and-memory/README.md) — backend/device selection messages surfaced through the log callback.
- [Text Generation & Sampling](../text-generation/README.md) — the inference loop whose performance/debug output you can capture.
- [Custom Builds & Platform Support](../custom-builds/README.md) — native library loading whose diagnostics flow through logging.
