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
 * Internal helpers shared by the high-level wrappers ({@link LlamaEmbedder},
 * {@link LlamaReranker}) to pick sensible defaults based on the GGUF model
 * architecture.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
final class TaskDefaults {

  private TaskDefaults() {}

  /**
   * Returns {@code true} when the given GGUF architecture string corresponds to a
   * bidirectional encoder (BERT-family) model. Encoder models use non-causal
   * attention and typically pool via CLS or MEAN.
   *
   * @param arch Value of the {@code general.architecture} GGUF metadata key
   * @return {@code true} for encoder architectures; {@code false} otherwise
   */
  static boolean isEncoderArch(String arch) {
    if (arch == null) {
      return false;
    }
    return (
      "bert".equals(arch) ||
      "nomic-bert".equals(arch) ||
      "nomic-bert-moe".equals(arch) ||
      "modern-bert".equals(arch) ||
      arch.startsWith("jina-bert") ||
      "neo-bert".equals(arch) ||
      "eurobert".equals(arch)
    );
  }
}
