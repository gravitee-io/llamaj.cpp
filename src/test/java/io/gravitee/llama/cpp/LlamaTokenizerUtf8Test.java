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

import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_TO_DOWNLOAD;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Regression test: llama_tokenize takes a BYTE length. Passing the Java string length
 * (UTF-16 chars) silently truncated the tail of any prompt containing multi-byte UTF-8
 * (observed as GLM-4 losing its trailing <|assistant|> marker and the end of the user
 * turn). The detokenized pieces of a multi-byte prompt must round-trip to the full text.
 */
class LlamaTokenizerUtf8Test extends LlamaCppTest {

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
    LlamaRuntime.llama_backend_free();
  }

  @Test
  void tokenize_must_not_truncate_multibyte_utf8_prompts() {
    var model = track(
      new LlamaModel(
        arena,
        getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD),
        new LlamaModelParams(arena)
      )
    );
    var context = track(
      new LlamaContext(arena, model, new LlamaContextParams(arena).nCtx(512))
    );
    var vocab = new LlamaVocab(model);
    var tokenizer = new LlamaTokenizer(vocab, context);

    String prompt =
      "Résumé — “quotes”, café, 東京, emoji 🚀. The tail must survive tokenization.";

    var resp = tokenizer.tokenize(arena, prompt);
    var bytes = new java.io.ByteArrayOutputStream();
    for (int i = 0; i < resp.size(); i++) {
      int id = resp.data().getAtIndex(JAVA_INT, i);
      bytes.writeBytes(vocab.tokenToPiece(id));
    }
    String roundTrip = new String(
      bytes.toByteArray(),
      java.nio.charset.StandardCharsets.UTF_8
    );

    // The BOS/leading special tokens may prepend text, but the FULL prompt — most
    // importantly its tail after the multi-byte characters — must be present.
    assertThat(roundTrip).endsWith("The tail must survive tokenization.");
    assertThat(roundTrip).contains("東京", "🚀", "café");
  }
}
