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

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import org.junit.jupiter.api.BeforeAll;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
abstract class LlamaCppTest {

    @BeforeAll
    public static void beforeAll() {
        LlamaLibLoader.load();
    }

    static final String MODEL_TO_DOWNLOAD = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-IQ3_M.gguf";
    static final String MODEL_PATH = "models/Llama-3.2-1B-Instruct-IQ3_M.gguf";
    static final String SYSTEM = """
            You are the best at guessing capitals. Respond to the best of your ability. Just answer with the capital.
            If asked for the capital of France, answer Paris.
            If asked for the UK or England, answer London.
            If asked for the capital of Poland, answer Warsaw.
            """;

    static Path getModelPath() {
        try {
            Path absolutePath = Path.of(MODEL_PATH).toAbsolutePath();
            if (!Files.exists(absolutePath)) {
                Files.copy(URI.create(MODEL_TO_DOWNLOAD).toURL().openStream(), absolutePath, REPLACE_EXISTING);
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
            LlamaContextParams contextParams) {
        var template = new LlamaTemplate(model);
        return template.applyTemplate(arena, messages, contextParams.nCtx());
    }

    static LlamaChatMessages buildMessages(Arena arena, String system, String input) {
        return new LlamaChatMessages(arena, List.of(
                new LlamaChatMessage(arena, Role.SYSTEM, system),
                new LlamaChatMessage(arena, Role.USER, input)
        ));
    }
}
