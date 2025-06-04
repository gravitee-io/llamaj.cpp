#  Llamaj.cpp

llamaj.cpp (contraction of llama.cpp and java/jextract) is a port of llama.cpp in the JVM using jextract.

## Requirements

- Java 21
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

## Build

1. Get `jextract`

> Make sure the `jextract` folder is in the same path level as your repository

On Linux:

Since we are using JDK 21 you can download a prebuilt version of `jextract`

```bash
$ wget https://download.java.net/java/early_access/jextract/21/1/openjdk-21-jextract+1-2_linux-x64_bin.tar.gz
$ tar -xzf openjdk-21-jextract+1-2_linux-x64_bin.tar.gz
$ rm openjdk-21-jextract+1-2_linux-x64_bin.tar.gz
$ echo 'export PATH="$(pwd)/jextract/bin:$PATH"' >> ~/.bashrc
```

On MacOS:
For JDK21, there is not a version of jextract for MacOS aarch64, only for x86_64, so we have to build it ourselves

```bash
$ git clone https://github.com/openjdk/jextract
$ cd jextract
$ git checkout jdk21
```

Make sure your `$JAVA_HOME` points to your jdk21

Since jextract for jdk21 uses gradle with a jdk17 version, we need to upgrade gradle version:
```bash
$ sed -i '' 's#gradle-7\.3\.3-bin\.zip#gradle-8.5-bin.zip#g' gradle/wrapper/gradle-wrapper.properties
```

Install llvm:
```bash
$ brew install llvm
```

Then execute the gradle command:
```bash
$ sh ./gradlew -Pjdk21_home=$JAVA_HOME -Pllvm_home=$(brew --prefix llvm) clean verify
```

Set `jextract` binaries to your path:
```bash
$ ln -sf $(pwd)/build/jextract/bin $(pwd)/bin
$ echo "PATH=$PATH:$(pwd)/bin" >> ~/.zshrc
$ source ~/.zshrc
```
3. Clone llama.cpp

> Make sure `llama.cpp` folder is in the same path level as your repository

```bash
$ git clone https://github.com/ggml-org/llama.cpp
```

5. Download binaries and generate the sources

```bash
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
$ java -jar llamaj.cpp-<version>.jar --model models/model.gguf --system 'You are a helpful assistant. Answer question to the best of your ability'
```

On linux, don't forget to link your libraries with the environment variable below:
```bash
$ mkdir $HOME/.llama.cpp
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
4.Add it to your classpath:`

gravitee-io/llamaj.cpp will pick up at runtime the os and architecture and will call the according bindings using reflection.
