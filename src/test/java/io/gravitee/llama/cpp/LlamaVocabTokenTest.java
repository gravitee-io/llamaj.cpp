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

import static io.gravitee.llama.cpp.LlamaCppTest.REASONING_MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.REASONNING_MODEL_TO_DOWNLOAD;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Integration tests for {@link LlamaVocab} special token accessors
 * and {@link LlamaTemplate} template string extraction.
 *
 * <p>Tests against Qwen3-0.6B GGUF to verify BOS/EOS token text
 * and chat template string retrieval from the model metadata.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@Tag("integration")
class LlamaVocabTokenTest extends LlamaCppTest {

  private static Arena arena;

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);
  }

  @AfterAll
  static void afterAll() {
    arena.close();
  }

  @Test
  void qwen3_vocab_has_eos_token() {
    Path modelPath = getModelPath(
      REASONING_MODEL_PATH,
      REASONNING_MODEL_TO_DOWNLOAD
    );
    var modelParams = new LlamaModelParams(arena);
    var model = new LlamaModel(arena, modelPath, modelParams);
    var vocab = new LlamaVocab(model);

    String bos = vocab.bosTokenText();
    String eos = vocab.eosTokenText();

    System.out.println("Qwen3 BOS: '" + bos + "'");
    System.out.println("Qwen3 EOS: '" + eos + "'");

    // Qwen3 has no BOS token (bos_token: null in tokenizer_config.json)
    // but does have EOS (<|im_end|> or <|endoftext|>)
    assertThat(eos).isNotEmpty();
  }

  @Test
  void qwen3_template_string_is_readable() {
    Path modelPath = getModelPath(
      REASONING_MODEL_PATH,
      REASONNING_MODEL_TO_DOWNLOAD
    );
    var modelParams = new LlamaModelParams(arena);
    var model = new LlamaModel(arena, modelPath, modelParams);

    var template = new LlamaTemplate(model);
    String templateString = template.templateString();

    System.out.println(
      "Qwen3 template length: " +
        (templateString != null ? templateString.length() : "null")
    );
    System.out.println(
      "Qwen3 template preview: " +
        (templateString != null
            ? templateString.substring(
              0,
              Math.min(200, templateString.length())
            )
            : "null")
    );

    assertThat(templateString).isNotNull();
    assertThat(templateString).isNotEmpty();
    assertThat(templateString).contains("{%");
    assertThat(templateString).contains("messages");
    // Qwen3 templates reference im_start/im_end
    assertThat(templateString).contains("im_start");
  }
}
