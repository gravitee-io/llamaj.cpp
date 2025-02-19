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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

import static io.gravitee.llama.cpp.llama_h_1.*;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class LlamaIteratorTest extends LlamaCppTest {

    static final String MODEL_TO_DOWNLOAD = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-IQ3_M.gguf";
    static final String MODEL_PATH = "models/Llama-3.2-1B-Instruct-IQ3_M.gguf";
    static final String SYSTEM = """
            You are the best at guessing capitals. Respond to the best of your ability. Just answer with the capital.
            """;
    static final int SEED = new Random().nextInt();

    static final String ENGLISH_GRAMMAR = """
            root        ::= en-char+ ([ \\t\\n] en-char+)*
            en-char     ::= letter | digit | punctuation
            letter      ::= [a-zA-Z]
            digit       ::= [0-9]
            punctuation ::= [!"#$%&'()*+,-./:;<=>?@[\\\\\\]^_`{|}~]
            """;

    static Stream<Arguments> params_that_allow_llama_generation() {
        return Stream.of(
                Arguments.of(SYSTEM, "What is the capital of France?", "Paris"),
                Arguments.of(SYSTEM, "What is the capital of the UK?", "London"),
                Arguments.of(SYSTEM, "What is the capital of Poland?", "Warsaw")
        );
    }

    @ParameterizedTest
    @MethodSource("params_that_allow_llama_generation")
    void llama_simple_generation(String system, String input, String expected) {


        try (Arena arena = Arena.ofConfined()) {
            var modelParameters = new LlamaModelParams(arena);
            Path absolutePath = getModelPath();
            var model = new LlamaModel(arena, absolutePath, modelParameters);
            var contextParams = new LlamaContextParams(arena);
            var context = new LlamaContext(model, contextParams);

            var vocab = new LlamaVocab(model);
            var sampler = new LlamaSampler(arena).seed(new Random().nextInt());

            var prompt = getPrompt(model, arena, builMessages(arena, system, input), contextParams);

            var it = new LlamaIterator(context, vocab, sampler, prompt);
            String output = "";
            for (; it.hasNext(); ) {
                output += it.next().content();
            }
            it.close();

            assertThat(it.getInputTokens()).isGreaterThan(0);
            assertThat(it.getInputTokens()).isGreaterThan(0);
            assertThat(output).containsIgnoringCase(expected);

            llama_sampler_free(sampler.segment);
            llama_free(context.segment);
            llama_model_free(model.segment);
        }
    }

    @ParameterizedTest
    @MethodSource("params_that_allow_llama_generation")
    void llama_tuned_generation(String system, String input, String expected) {


        try (Arena arena = Arena.ofConfined()) {
            var modelParameters = new LlamaModelParams(arena);
            Path absolutePath = getModelPath();
            var model = new LlamaModel(arena, absolutePath, modelParameters);
            var contextParams = new LlamaContextParams(arena)
                    .nCtx(512)
                    .nBatch(512)
                    .nThreads(16)
                    .nThreadsBatch(16)
                    .attentionType(AttentionType.CAUSAL)
                    .embeddings(false)
                    .offloadKQV(true)
                    .flashAttn(true)
                    .noPerf(true);

            var context = new LlamaContext(model, contextParams);

            var vocab = new LlamaVocab(model);
            var sampler = new LlamaSampler(arena).seed(new Random().nextInt())
                    .seed(SEED)
                    .temperature(0.75f)
                    .topK(40)
                    .topP(0.2f, 40)
                    .minP(0.05f, 40)
                    .mirostat(SEED, 3, 0.1f)
                    .grammar(vocab, ENGLISH_GRAMMAR, "root")
                    .penalties(10, 1.0f, 0.3f, 0.0f);

            var prompt = getPrompt(model, arena, builMessages(arena, system, input), contextParams);

            var it = new LlamaIterator(context, vocab, sampler, prompt);
            String output = "";
            for (; it.hasNext(); ) {
                output += it.next().content();
            }
            it.close();

            assertThat(it.getInputTokens()).isGreaterThan(0);
            assertThat(it.getInputTokens()).isGreaterThan(0);
            assertThat(output).containsIgnoringCase(expected);

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

    private static String getPrompt(
            LlamaModel model,
            Arena arena,
            LlamaChatMessages messages,
            LlamaContextParams contextParams) {
        var template = new LlamaTemplate(model);
        return template.applyTemplate(arena, messages, contextParams.nCtx());
    }

    private static LlamaChatMessages builMessages(Arena arena, String system, String input) {
            return new LlamaChatMessages(arena, List.of(
                    new LlamaChatMessage(arena, Role.SYSTEM, system),
                    new LlamaChatMessage(arena, Role.USER, input)
            ));
        }
}
