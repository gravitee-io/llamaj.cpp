# Gravitee Llama.cpp

A port of llama.cpp in the JVM using jextract.

## Requirements

- Java 21
- mvn
- MacOS Mseries (Linux coming soon)

## Build

```bash
$ mvn clean install
```

## Run

```bash
$ java --enable-preview \
     --enable-native-access=ALL-UNNAMED \
     -Dfile.encoding=UTF-8 \
     -Dsun.stdout.encoding=UTF-8 \
     -Dsun.stderr.encoding=UTF-8 \
     -classpath target/classes:gravitee-llama-cpp.jar \
     io.gravitee.llama.cpp.Main \
     src/main/resources/libllama.dylib /path/to/your/model/model.gguf
```

There are plenty of models on HuggingFace, we suggest the one [here](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) 