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

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
abstract class LlamaCppTest {

  static final String MODEL_TO_DOWNLOAD =
    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-IQ3_M.gguf";
  static final String REASONNING_MODEL_TO_DOWNLOAD =
    "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf";

  static final String MODEL_PATH = "models/model.gguf";
  static final String REASONING_MODEL_PATH = "models/reasoning.gguf";

  static final String SYSTEM = """
     You are the best at guessing capitals. Respond to the best of your ability. Just answer with the capital.
     What's the capital of France ? Paris.
     What is the capital of England? London.
    What is the capital of Poland? Warsaw.
     """;

  static final String VL_TEXT =
    "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q8_0.gguf";
  static final String MODEL_VL_PATH = "models/model_vl.gguf";

  static final String VL_MMPROJ =
    "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf";
  static final String VL_MMPROJ_PATH = "models/mmproj.gguf";

  // Audio model paths (Ultravox for audio understanding)
  static final String AUDIO_MODEL_TO_DOWNLOAD =
    "https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf";
  static final String AUDIO_MODEL_PATH = "models/model_audio.gguf";

  // Audio mmproj
  static final String AUDIO_MMPROJ_TO_DOWNLOAD =
    "https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/resolve/main/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf";
  static final String AUDIO_MMPROJ_PATH = "models/mmproj_audio.gguf";

  static final String LORA_ADAPTER_TO_DOWNLOAD =
    "https://huggingface.co/bunnycore/LLama-3.2-1B-General-lora_model-F16-GGUF/resolve/main/LLama-3.2-1B-General-lora_model-f16.gguf";
  static final String LORA_ADATAPTER_PATH = "models/lora-adapter.gguf";

  static Path getModelPath(String modelPath, String modelToDownload) {
    return getPath(modelPath, modelToDownload);
  }

  private static Path getPath(
    String loraAdatapterPath,
    String loraAdapterToDownload
  ) {
    try {
      Path absolutePath = Path.of(loraAdatapterPath).toAbsolutePath();
      if (!Files.exists(absolutePath)) {
        Files.copy(
          URI.create(loraAdapterToDownload).toURL().openStream(),
          absolutePath,
          REPLACE_EXISTING
        );
      }
      return absolutePath;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  static String getPrompt(
    LlamaModel model,
    Arena arena,
    LlamaChatMessages messages,
    LlamaContextParams contextParams
  ) {
    var template = new LlamaTemplate(model);
    return template.applyTemplate(arena, messages, contextParams.nCtx());
  }

  static LlamaChatMessages buildMessages(
    Arena arena,
    String system,
    String input
  ) {
    return new LlamaChatMessages(
      arena,
      List.of(
        new LlamaChatMessage(arena, Role.SYSTEM, system),
        new LlamaChatMessage(arena, Role.USER, input)
      )
    );
  }
}
