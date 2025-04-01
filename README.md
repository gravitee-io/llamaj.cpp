# Gravitee Llama.cpp

A port of llama.cpp in the JVM using jextract.

## Requirements

- Java 21
- mvn
- MacOS M-series / Linux x86_64 (CPU)

## Build

```bash
$ mvn clean install
```

## Run

```bash
$ mvn exec:java -Dexec.mainClass=io.gravitee.llama.cpp.Main \
    -Dexec.args="/path/to/model/ 'You are a helpful assistant. Answer question to the best of your ability'"
```

On linux don't forget to link your libraries with the environment variable below:
```bash
$ export LD_LIBRARY_PATH="$HOME/.llama.cpp:$LD_LIBRARY_PATH"
```
`
There are plenty of models on HuggingFace, we suggest the one [here](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)

## Use your own llama.cpp build

1. Clone `llama.cpp` repository

```bash
$ git clone https://github.com/ggml-org/llama.cpp
$ cd llama.cpp
```

2. Compile sources

Make sure you have gcc / g++ compiler

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
The path temporary file will be used to loaad the shared object libraries

## Build jextract sources yourself

1. Get `jextract`

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
$ echo "PATH=$PATH:$(pwd)/bin" >> ~/.zshrc
$ source ~/.zshrc
```

3. Generate the sources

4. Go to your `llama.cpp` path and execute:

For Linux x86_64
```bash
$ jextract -t io.gravitee.llama.cpp.linux.x86_64 \
           --source \
           --include-dir ggml/include \
           --output /path/to/gravitee-llama-cpp/src/main/java/ include/llama.h
```
For MacOs aarch64
```bash
$ jextract -t io.gravitee.llama.cpp.macosx.aarch64 \
           --source \
           --include-dir ggml/include \
           --output /path/to/gravitee-llama-cpp/src/main/java/ include/llama.h
```
