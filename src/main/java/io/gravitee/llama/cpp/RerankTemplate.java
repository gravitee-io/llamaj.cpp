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

/**
 * Formats a (query, document) pair into the exact input string to feed to a
 * cross-encoder reranker model.
 *
 * <p>Different reranker families require different input formats:
 * <ul>
 *   <li>BERT cross-encoders (BGE-reranker, Jina-reranker) accept plain
 *       concatenation; the BERT tokenizer auto-inserts the [CLS] and [SEP]
 *       special tokens when {@code add_special=true}. Use {@link #PLAIN}.</li>
 *   <li>Qwen3-Reranker and other chat-style rerankers require a structured
 *       prompt wrapping the query and document inside system/user messages.
 *       Pass a custom lambda or implementation, or delegate templating to an
 *       external library such as gravitee-inference.</li>
 * </ul>
 *
 * <p>This is a functional interface; any {@code BiFunction}-like lambda works:
 * <pre>{@code
 * RerankTemplate plain = (q, d) -> q + " " + d;
 *
 * RerankTemplate qwen3 = (q, d) ->
 *   "<|im_start|>system\nJudge relevance...<|im_end|>\n" +
 *   "<|im_start|>user\nQuery: " + q + "\nDocument: " + d + "\nRelevant:<|im_end|>\n";
 * }</pre>
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@FunctionalInterface
public interface RerankTemplate {
  /**
   * Builds the tokeniser input for the given (query, document) pair.
   *
   * @param query    The search query
   * @param document The candidate document
   * @return The formatted input string to pass to the tokenizer
   */
  String format(String query, String document);

  /**
   * Plain space-separated concatenation: {@code query + " " + document}.
   * Suitable for BERT-family cross-encoders where the tokenizer inserts
   * the CLS / SEP special tokens automatically.
   */
  RerankTemplate PLAIN = (query, document) -> query + " " + document;
}
