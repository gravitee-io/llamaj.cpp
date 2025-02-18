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

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

import static io.gravitee.llama.cpp.llama_h_1.*;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class LlamaIteratorTest extends LlamaCppTest {

    private static final String MODEL_TO_DOWNLOAD = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-IQ3_M.gguf";
    public static final String MODEL_PATH = "models/Llama-3.2-1B-Instruct-IQ3_M.gguf";

    @Test
    void llama_simple_generation() {

        ggml_backend_load_all();

        try (Arena arena = Arena.ofConfined()) {

            var modelParameters = new LlamaModelParams(arena);
            Path absolutePath = getModelPath();
            var model = new LlamaModel(arena, absolutePath, modelParameters);
            var contextParams = new LlamaContextParams(arena).nSeqMax(10);
            var context = new LlamaContext(model, contextParams);

            var vocab = new LlamaVocab(model);
            var sampler = new LlamaSampler(arena).seed(new Random().nextInt());

            var prompt = getPrompt(model, arena, contextParams);

            var it = new LlamaIterator(context, vocab, sampler, prompt);
            String output = "";
            for (; it.hasNext(); ) {
                output += it.next().token();
            }
            it.close();

            assertThat(it.getInputTokens()).isGreaterThan(0);
            assertThat(it.getInputTokens()).isGreaterThan(0);
            assertThat(output).containsIgnoringCase("paris");

            llama_sampler_free(sampler.segment);
            llama_free(context.segment);
            llama_model_free(model.segment);
        }
    }

    private static Path getModelPath() {
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

    private static String getPrompt(LlamaModel model, Arena arena, LlamaContextParams contextParams) {
        var template = new LlamaTemplate(model);
        var messages = builMessages(arena);

        return template.applyTemplate(arena, messages, contextParams.nCtx());
    }

    private static LlamaChatMessages builMessages(Arena arena) {
        return new LlamaChatMessages(arena, List.of(
                new LlamaChatMessage(arena, Role.SYSTEM, """
                        You are the best at guessing capitals. Respond to the best of your ability. Just answer with the capital
                        """
                ),
                new LlamaChatMessage(arena, Role.USER, "What is the capital of France?")
        ));
    }


}
