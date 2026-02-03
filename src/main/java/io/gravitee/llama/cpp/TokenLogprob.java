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
 * Represents the log-probability of a single token.
 *
 * <p>This mirrors the structure used by OpenAI-compatible logprobs APIs:
 * each token carries its decoded text, the log-probability assigned by
 * the model, and the raw UTF-8 bytes of the token piece (useful for
 * reassembling multi-byte characters split across tokens).
 *
 * @param token   The decoded text of the token (may be empty for special tokens)
 * @param tokenId The vocabulary ID of the token
 * @param logprob The log-probability (ln(p)) of this token; will be
 *                {@link Double#NEGATIVE_INFINITY} if the model assigned
 *                probability zero to this token
 * @param bytes   The raw UTF-8 bytes of the token piece, or an empty list
 *                if the token has no byte representation
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record TokenLogprob(
  String token,
  int tokenId,
  double logprob,
  List<Integer> bytes
) {}
