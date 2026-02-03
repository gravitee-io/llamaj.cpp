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

import java.util.List;

/**
 * Log-probability information for a single generated token position.
 *
 * <p>Contains the log-probability of the token that was actually sampled
 * ({@code chosenToken}), along with the {@code topLogprobs} most-likely
 * alternative tokens at this position, sorted by descending log-probability.
 *
 * <p>This is analogous to the per-token logprobs object returned by
 * OpenAI-compatible chat-completion APIs when {@code logprobs=true}.
 *
 * <p>If logprobs collection was not enabled for this generation (i.e. the
 * {@link ConversationState} was not configured with
 * {@link ConversationState#setTopLogprobs(int)}, this field will be
 * {@code null} in the corresponding {@link LlamaOutput}.
 *
 * @param chosenToken The token that was actually sampled at this position
 * @param topLogprobs The top-N candidate tokens with their log-probabilities,
 *                    sorted by descending log-probability.  The chosen token
 *                    is always included in this list.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record Logprobs(
  TokenLogprob chosenToken,
  List<TokenLogprob> topLogprobs
) {}
