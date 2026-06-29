# Custom Builds & Platform Support

> Build the llama.cpp native libraries and regenerate the jextract FFM bindings for platforms beyond the two shipped out of the box (macOS Apple Silicon and Linux x86_64).

## Overview
llamaj.cpp ships pre-built bindings and native libraries for **macOS `aarch64`** and **Linux `x86_64`** only, selected at build time by a Maven profile. To run on anything else (Windows, ARM Linux, x86_64 macOS, or a CUDA / OpenBLAS / AVX-512 build), you compile llama.cpp yourself, regenerate the Java FFM bindings with `jextract`, fix up a couple of generated files, package them into a JAR, and drop the matching native libraries where `LlamaLibLoader` can find them. The runtime is platform-agnostic: `OperatingSystem` already includes `WINDOWS` and `LlamaLibLoader` already knows how to load `.dll`/`.so`/`.dylib` — the only missing pieces for a new target are the generated bindings and the native libs.

## Key types
- `LlamaLibLoader` — resolves and `System.load()`s the native libraries; `load()` (auto-detect) and `load(String path)` return the directory that was loaded.
- `Platform` / `PlatformResolver` — `PlatformResolver.platform()` detects OS + arch; `Platform.getPackage()` returns the dotted name (e.g. `windows.x86_64`) used for **both** the jextract target package and the native-lib resource folder.
- `OperatingSystem` — supported OS values: `MAC_OS_X` (`macosx`), `LINUX` (`linux`), `WINDOWS` (`windows`).
- `Architecture` — supported arch values: `X86_64` (`x86_64`), `AARCH64` (`aarch64`).
- `LlamaRuntime` — backend lifecycle: `llama_backend_init()`, `ggml_backend_load_all_from_path(arena, libPath)`, `llama_backend_free()`.

## Usage

### Shipped platforms — one Maven command
The `macosx-aarch64` / `linux-x86_64` profiles download the right `jextract`, download the matching pre-built llama.cpp release (pinned by the `llama.cpp.version` property, currently `b9673`), run `jextract`, post-process, format, and install the artifact:

```bash
# macOS (Apple Silicon)
mvn prettier:write license:format clean generate-sources -Pmacosx-aarch64 install

# Linux (x86_64) — then point the loader at the libs at runtime
mvn prettier:write license:format clean generate-sources -Plinux-x86_64 install
export LD_LIBRARY_PATH="$HOME/.llama.cpp:$LD_LIBRARY_PATH"
```

### A new platform (e.g. Windows x86_64) — manual pipeline

```bash
# 1. Build llama.cpp for the target (add -DGGML_CUDA=ON etc. as needed — see llama.cpp/docs/build.md)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release

# 2. Generate the FFM bindings with jextract (from OpenJDK EA build 25/2).
#    The target package MUST be io.gravitee.llama.cpp.<os>.<arch> — it has to match
#    Platform.getPackage() for the detected platform (here: windows.x86_64).
jextract -t io.gravitee.llama.cpp.windows.x86_64 \
  --include-dir /path/to/llama.cpp/ggml/include \
  --include-dir /path/to/llama.cpp/include \
  --output src/main/java \
  --header-class-name llama_h \
  /path/to/llama.cpp/tools/mtmd/mtmd.h \
  /path/to/llama.cpp/tools/mtmd/mtmd-helper.h \
  /path/to/llama.cpp/include/llama.h \
  /path/to/llama.cpp/ggml/include/ggml-rpc.h

# 3. Post-process the generated sources (see Notes): make llama_h_1 / llama_h_2 public,
#    and fully-qualify ggml_backend_graph_copy.layout() in the new package.

# 4. Compile + package the generated sources into a bindings JAR (Maven/Gradle/javac).

# 5. Place the native libraries (.dll for Windows) under the resource folder that
#    matches the package — src/main/resources/windows/x86_64/ — so they get packaged
#    into the JAR and resolved by LlamaLibLoader at runtime.
```

### Consuming a custom build at runtime
This mirrors every integration test (`LlamaLibLoader.load()` then `ggml_backend_load_all_from_path`):

```java
import io.gravitee.llama.cpp.*;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;

var arena = Arena.ofConfined();

// Resolve + System.load() the native libs, then init the backend and load plugins.
String libPath = LlamaLibLoader.load();              // returns the loaded directory
LlamaRuntime.llama_backend_init();
LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

// ... load model / context / sampler exactly as on a shipped platform ...

LlamaRuntime.llama_backend_free();
arena.close();
```

Point the loader at libraries built outside the JAR by exporting `LLAMA_CPP_LIB_PATH`:

```bash
export LLAMA_CPP_LIB_PATH=/path/to/llama.cpp/build/bin
java --enable-preview --enable-native-access=ALL-UNNAMED -jar your-app.jar ...
```

## Options

### Helper scripts (used by the Maven profiles, runnable standalone)
| Script | Purpose |
| --- | --- |
| `scripts/download-jextract.sh -o <os> -p <platform> [-d <dir>]` | Download the matching `jextract` EA build into `.jextract/` (only `macosx/aarch64`, `linux/x86_64` are wired up). |
| `scripts/download-native-libraries.sh -o <os> -p <platform> -v <version> -d <dest>` | Download a pre-built llama.cpp release and flatten the `.so`/`.dylib` files into `<dest>/<os>/<platform>`. |

### `jextract` flags (per the README / pom invocation)
| Flag | Value |
| --- | --- |
| `-t` | Target package `io.gravitee.llama.cpp.<os>.<arch>` (must equal `Platform.getPackage()`). |
| `--include-dir` | `…/ggml/include` and `…/include` from your llama.cpp checkout. |
| `--header-class-name` | `llama_h`. |
| `--output` | Source output dir (e.g. `src/main/java` or `target/generated-sources`). |
| headers | `mtmd.h`, `mtmd-helper.h`, `llama.h`, `ggml-rpc.h`. |

### Native-library resolution (`LlamaLibLoader`)
| Mechanism | Effect |
| --- | --- |
| `LLAMA_CPP_LIB_PATH` (env var) | Load `.so`/`.dylib`/`.dll` directly from this directory; takes priority over bundled resources. |
| `-DLLAMA_CPP_USE_TMP_LIB_PATH=true` (JVM system property) | Copy the libraries into a fresh temp dir before loading instead of caching in `~/.llama.cpp`. |
| (default) | Extract the libs bundled under the `Platform.getPackage()` resource folder into `~/.llama.cpp` and load from there. |

## Notes
- **Target package is load-bearing.** `jextract -t` and the native-lib folder must both equal `Platform.getPackage()` (`<osName>.<arch>`, e.g. `windows.x86_64`). At runtime `LlamaLibLoader` looks for native libs under exactly that package (`os.getOsName() + "." + arch.getArch()`); a mismatch means the libs are never found.
- **Post-processing is required** because jextract emits package-private overflow classes. The shipped profiles: (1) rewrite `class llama_h_1 …`/`class llama_h_2 …` to `public class …`, and (2) fully-qualify `ggml_backend_graph_copy.layout()` to `io.gravitee.llama.cpp.<os>.<arch>.ggml_backend_graph_copy.layout()`. Re-apply the equivalent fixes for your generated package, then verify the sources compile.
- **`jextract` versions are pinned** to the OpenJDK 25/2 early-access build; use the same major to keep the generated `MethodHandle`/`MemoryLayout` API compatible with the hand-written wrappers. Requires **Java 25** and `--enable-preview --enable-native-access=ALL-UNNAMED` at runtime.
- **The loader already supports Windows** (`.dll`) and arbitrary arches at the Java level — `OperatingSystem.WINDOWS` and the `DLL_EXT` branch exist. Only the generated bindings + native binaries are missing for an unsupported target.
- **Linux plugin layout.** On Linux, `LlamaLibLoader` deliberately loads only soname-versioned files (`libfoo.so.N`) and skips backend plugins (`libggml-cpu/cuda/…`); those are dlopen'd by `ggml_backend_load_all_from_path`. Keep the full release layout (don't rename/strip versions) so the backend registry resolves to a single instance. Set `LD_LIBRARY_PATH=$HOME/.llama.cpp` so NEEDED dependencies resolve.
- **Building with accelerators** (CUDA, OpenBLAS, AVX2/AVX-512, Vulkan, …) is a llama.cpp cmake concern — pass the relevant `-D…=ON` flags; see llama.cpp's `docs/build.md`. The bindings themselves do not change, only the native libraries you ship.
- **Resource lifecycle** is identical to the shipped platforms: `model.free()`, `context.free()`, `sampler.free()`, then `LlamaRuntime.llama_backend_free()` and `arena.close()`.

## See also
- [Getting Started](../getting-started/README.md) — first run on a shipped platform; backend init and `LlamaLibLoader.load()`.
- [Distributed Inference (RPC)](../distributed-inference/README.md) — uses the `ggml-rpc.h` binding that this build includes.
- [Devices & Memory](../device-and-memory/README.md) — selecting backends/devices once the native libs are loaded.
- [Logging](../logging/README.md) — surface native-library load and backend-registration diagnostics.
