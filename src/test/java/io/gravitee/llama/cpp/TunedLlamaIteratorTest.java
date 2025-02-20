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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static io.gravitee.llama.cpp.llama_h_1.*;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class TunedLlamaIteratorTest extends LlamaCppTest {

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

    static Arena arena;
    static LlamaModel model;
    static LlamaVocab vocab;
    static LlamaContextParams contextParams;
    static LlamaContext context;
    static LlamaSampler sampler;

    @BeforeAll
    public static void beforeAll() throws IOException {
        LlamaCppTest.beforeAll();
        ggml_backend_load_all();

        arena = Arena.ofAuto();

        var logger = new LlamaLogger(arena);
        logger.setLogging(LlamaLogLevel.ERROR);

        var modelParameters = new LlamaModelParams(arena);
        Path absolutePath = getModelPath();
        model = new LlamaModel(arena, absolutePath, modelParameters);

        contextParams = new LlamaContextParams(arena)
                .nCtx(512)
                .nBatch(512)
                .nThreads(16)
                .nThreadsBatch(16)
                .attentionType(AttentionType.CAUSAL)
                .embeddings(false)
                .offloadKQV(true)
                .flashAttn(true)
                .noPerf(true);
        context = new LlamaContext(model, contextParams);

        vocab = new LlamaVocab(model);
        sampler = new LlamaSampler(arena)
                .seed(SEED)
                .temperature(0.75f)
                .topK(40)
                .topP(0.2f, 40)
                .minP(0.05f, 40)
                .mirostat(SEED, 3, 0.1f)
                .grammar(vocab, ENGLISH_GRAMMAR, "root")
                .penalties(10, 1.0f, 0.3f, 0.0f);
    }

    @ParameterizedTest
    @MethodSource("params_that_allow_llama_generation")
    void llama_tuned_generation(String system, String input, String expected) {
        var prompt = getPrompt(model, arena, builMessages(arena, system, input), contextParams);

        var it = new LlamaIterator(context, vocab, sampler, prompt)
                .setStopStrings(List.of("."))
                .setQuota(10);
        String output = "";
        for (; it.hasNext(); ) {
            output += it.next().content();
        }
        it.close();

        assertThat(it.getInputTokens()).isGreaterThan(0);
        assertThat(it.getOutputTokens()).isGreaterThan(0);
        assertThat(output).containsIgnoringCase(expected);
        System.out.println(output);
    }

    @AfterAll
    public static void afterAll() {
        llama_sampler_free(sampler.segment);
        llama_free(context.segment);
        llama_model_free(model.segment);
        arena = null;
    }
}
