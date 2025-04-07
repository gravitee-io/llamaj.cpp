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

import java.lang.foreign.Arena;
import java.util.stream.Stream;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class LlamaChatMessageTest extends LlamaCppTest{

    public static Stream<Arguments> params_that_must_init_message() {
        return Stream.of(
                Arguments.of(Role.SYSTEM, "You are a helpful assistant"),
                Arguments.of(Role.USER, "What's the capital of France?"),
                Arguments.of(Role.ASSISTANT, "Paris")
        );
    }

    @ParameterizedTest
    @MethodSource("params_that_must_init_message")
    void must_init_message(Role role, String content) {
        try (Arena arena = Arena.ofConfined()){
            var message = new LlamaChatMessage(arena, role, content);

            assertThat(message.getRole()).isEqualTo(role);
            assertThat(message.getContent()).isEqualTo(content);
        }
    }
}
