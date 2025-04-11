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

import io.gravitee.llama.cpp.platform.PlatformResolver;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaRuntime {

  private static final String runtime = PlatformResolver.platform().runtime();
  public static final String MACOSX_AARCH_64 = "macosx_aarch64";
  public static final String LINUX_X86_64 = "linux_x86_64";

  private LlamaRuntime() {}

  /* load backends */
  public static void ggml_backend_load_all() {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.ggml_backend_load_all();
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.ggml_backend_load_all();
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  /* logging */
  static void llama_log_set(MemorySegment m1, MemorySegment m2) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_log_set(m1, m2);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_log_set(m1, m2);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  /* Model Parameters */
  static MemorySegment llama_model_params_ofAddress(MemorySegment segment, Arena arena) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.ofAddress(segment, arena);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.ofAddress(segment, arena);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_model_default_params(Arena arena) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_model_default_params(arena);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_model_default_params(arena);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static long llama_max_devices() {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_max_devices();
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_max_devices();
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static int n_gpu_layers(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.n_gpu_layers$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.n_gpu_layers$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void n_gpu_layers(MemorySegment segment, int layers) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.n_gpu_layers$set(segment, layers);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.n_gpu_layers$set(segment, layers);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static int split_mode(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.split_mode$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.split_mode$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void split_mode(MemorySegment segment, int ordinal) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.split_mode$set(segment, ordinal);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.split_mode$set(segment, ordinal);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static int main_gpu(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.main_gpu$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.main_gpu$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void main_gpu(MemorySegment segment, int mainGpu) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.main_gpu$set(segment, mainGpu);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.main_gpu$set(segment, mainGpu);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static MemorySegment tensor_split(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.tensor_split$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.tensor_split$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void tensor_split(MemorySegment segment, MemorySegment tensorSplit) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.tensor_split$set(segment, tensorSplit);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.tensor_split$set(segment, tensorSplit);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static boolean vocab_only(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.vocab_only$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.vocab_only$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void vocab_only(MemorySegment segment, boolean vocabOnly) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.vocab_only$set(segment, vocabOnly);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.vocab_only$set(segment, vocabOnly);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static boolean use_mmap(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.use_mmap$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.use_mmap$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void use_mmap(MemorySegment segment, boolean useMmap) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.use_mmap$set(segment, useMmap);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.use_mmap$set(segment, useMmap);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static boolean use_mlock(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.use_mlock$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.use_mlock$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void use_mlock(MemorySegment segment, boolean useMlock) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.use_mlock$set(segment, useMlock);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.use_mlock$set(segment, useMlock);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static boolean check_tensors(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.check_tensors$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.check_tensors$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void check_tensors(MemorySegment segment, boolean checkTensors) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_model_params.check_tensors$set(
        segment,
        checkTensors
      );
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_model_params.check_tensors$set(segment, checkTensors);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  /* Model */
  static MemorySegment llama_model_load_from_file(MemorySegment modelPath, MemorySegment modelParams) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_model_load_from_file(
        modelPath,
        modelParams
      );
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_model_load_from_file(modelPath, modelParams);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  /* Vocab */
  static MemorySegment llama_model_get_vocab(MemorySegment model) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_model_get_vocab(model);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_model_get_vocab(model);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static boolean llama_vocab_is_eog(MemorySegment vocab, int tokenId) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_vocab_is_eog(vocab, tokenId);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_vocab_is_eog(vocab, tokenId);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static int llama_token_to_piece(
    MemorySegment vocab,
    int token,
    MemorySegment buf,
    int length,
    int lstrip,
    boolean special
  ) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_token_to_piece(
        vocab,
        token,
        buf,
        length,
        lstrip,
        special
      );
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_token_to_piece(
        vocab,
        token,
        buf,
        length,
        lstrip,
        special
      );
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  /* Sampler */

  static MemorySegment llama_sampler_chain_init(MemorySegment sampler) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_chain_init(sampler);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_chain_init(sampler);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_sampler_chain_default_params(SegmentAllocator allocator) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_chain_default_params(allocator);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_chain_default_params(allocator);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static int llama_sampler_sample(MemorySegment sampler, MemorySegment context, int idx) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_sample(sampler, context, idx);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_sample(sampler, context, idx);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void llama_sampler_chain_add(MemorySegment sampler, MemorySegment config) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_chain_add(sampler, config);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_chain_add(sampler, config);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static MemorySegment llama_sampler_init_temp(float temperature) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_init_temp(temperature);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_init_temp(temperature);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_sampler_init_top_k(int topK) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_init_top_k(topK);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_init_top_k(topK);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_sampler_init_top_p(float topP, int minKeep) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_init_top_p(topP, minKeep);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_init_top_p(topP, minKeep);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_sampler_init_min_p(float minP, int minKeep) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_init_min_p(minP, minKeep);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_init_min_p(minP, minKeep);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_sampler_init_mirostat_v2(int seed, float tau, float eta) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_init_mirostat_v2(seed, tau, eta);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_init_mirostat_v2(seed, tau, eta);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_sampler_init_grammar(MemorySegment vocab, MemorySegment grammar, MemorySegment root) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_init_grammar(vocab, grammar, root);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_init_grammar(vocab, grammar, root);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_sampler_init_penalties(
    int penaltyLastN,
    float penaltyRepeat,
    float penaltyFreq,
    float penaltyPresent
  ) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_init_penalties(
        penaltyLastN,
        penaltyRepeat,
        penaltyFreq,
        penaltyPresent
      );
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_init_penalties(
        penaltyLastN,
        penaltyRepeat,
        penaltyFreq,
        penaltyPresent
      );
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_sampler_init_dist(int seed) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_init_dist(seed);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_init_dist(seed);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  /* Context Params */
  static MemorySegment llama_context_default_params(SegmentAllocator allocator) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_context_default_params(allocator);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_context_default_params(allocator);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static int n_ctx(MemorySegment contextParams) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_ctx$get(contextParams);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_ctx$get(contextParams);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void n_ctx(MemorySegment segment, int nCtx) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_ctx$set(segment, nCtx);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_ctx$set(segment, nCtx);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static int n_batch(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_batch$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_batch$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void n_batch(MemorySegment segment, int nBatch) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_batch$set(segment, nBatch);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_batch$set(segment, nBatch);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static int n_ubatch(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_ubatch$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_ubatch$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void n_ubatch(MemorySegment segment, int nUBatch) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_ubatch$set(segment, nUBatch);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_ubatch$set(segment, nUBatch);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static int n_seq_max(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_seq_max$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_seq_max$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void n_seq_max(MemorySegment segment, int nSeqMax) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_seq_max$set(segment, nSeqMax);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_seq_max$set(segment, nSeqMax);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static int n_threads(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_threads$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_threads$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void n_threads(MemorySegment segment, int nThreads) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_threads$set(segment, nThreads);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_threads$set(segment, nThreads);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static int n_threads_batch(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_threads_batch$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_threads_batch$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void n_threads_batch(MemorySegment segment, int nThreadBatch) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.n_threads_batch$set(
        segment,
        nThreadBatch
      );
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.n_threads_batch$set(
        segment,
        nThreadBatch
      );
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static int pooling_type(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.pooling_type$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.pooling_type$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void pooling_type(MemorySegment segment, int ordinal) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.pooling_type$set(segment, ordinal);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.pooling_type$set(segment, ordinal);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static int attention_type(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.attention_type$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.attention_type$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void attention_type(MemorySegment segment, int ordinal) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.attention_type$set(segment, ordinal);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.attention_type$set(segment, ordinal);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static boolean embeddings(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.embeddings$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.embeddings$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void embeddings(MemorySegment segment, boolean embeddings) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.embeddings$set(segment, embeddings);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.embeddings$set(segment, embeddings);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static boolean offload_kqv(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.offload_kqv$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.offload_kqv$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void offload_kqv(MemorySegment segment, boolean offloadKQV) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.offload_kqv$set(segment, offloadKQV);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.offload_kqv$set(segment, offloadKQV);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static boolean flash_attn(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.flash_attn$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.flash_attn$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void flash_attn(MemorySegment segment, boolean flashAttn) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.flash_attn$set(segment, flashAttn);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.flash_attn$set(segment, flashAttn);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static boolean no_perf(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.no_perf$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.no_perf$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void no_perf(MemorySegment segment, boolean flashAttn) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_context_params.no_perf$set(segment, flashAttn);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_context_params.no_perf$set(segment, flashAttn);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  /* Context */
  static MemorySegment llama_init_from_model(MemorySegment model, MemorySegment params) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_init_from_model(model, params);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_init_from_model(model, params);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void llama_kv_cache_clear(MemorySegment context) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_kv_cache_clear(context);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_kv_cache_clear(context);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static int llama_get_kv_cache_used_cells(MemorySegment context) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_get_kv_cache_used_cells(context);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_get_kv_cache_used_cells(context);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  public static void llama_kv_cache_seq_rm(MemorySegment ctx, int seq_id, int p0, int p1) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_kv_cache_seq_rm(ctx, seq_id, p0, p1);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_kv_cache_seq_rm(ctx, seq_id, p0, p1);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  public static void llama_kv_cache_seq_add(MemorySegment ctx, int seq_id, int p0, int p1, int delta) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_kv_cache_seq_add(
        ctx,
        seq_id,
        p0,
        p1,
        delta
      );
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_kv_cache_seq_add(ctx, seq_id, p0, p1, delta);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  /* Chat Template */
  static MemorySegment llama_model_chat_template(MemorySegment model, MemorySegment name) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_model_chat_template(model, name);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_model_chat_template(model, name);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static int llama_chat_apply_template(
    MemorySegment tmpl,
    MemorySegment chat,
    long n_msg,
    boolean add_ass,
    MemorySegment buf,
    int length
  ) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_chat_apply_template(
        tmpl,
        chat,
        n_msg,
        add_ass,
        buf,
        length
      );
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_chat_apply_template(
        tmpl,
        chat,
        n_msg,
        add_ass,
        buf,
        length
      );
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  /* Chat Message */
  static MemorySegment llama_chat_message_allocate(SegmentAllocator allocator) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_chat_message.allocate(allocator);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_chat_message.allocate(allocator);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_chat_message_allocateArray(int size, SegmentAllocator allocator) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_chat_message.allocateArray(size, allocator);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_chat_message.allocateArray(size, allocator);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static MemorySegment llama_chat_message_role(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_chat_message.role$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_chat_message.role$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void llama_chat_message_role(MemorySegment message, MemorySegment role) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_chat_message.role$set(message, role);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_chat_message.role$set(message, role);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static MemorySegment llama_chat_message_content(MemorySegment segment) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_chat_message.content$get(segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_chat_message.content$get(segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static void llama_chat_message_content(MemorySegment message, MemorySegment content) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_chat_message.content$set(message, content);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_chat_message.content$set(message, content);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static long llama_chat_message_sizeof() {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_chat_message.sizeof();
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_chat_message.sizeof();
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  /* Tokenizer */
  static int llama_tokenize(
    MemorySegment vocab,
    MemorySegment text,
    int text_len,
    MemorySegment tokens,
    int n_tokens_max,
    boolean add_special,
    boolean parse_special
  ) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_tokenize(
        vocab,
        text,
        text_len,
        tokens,
        n_tokens_max,
        add_special,
        parse_special
      );
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_tokenize(
        vocab,
        text,
        text_len,
        tokens,
        n_tokens_max,
        add_special,
        parse_special
      );
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  /* Batch */
  static MemorySegment llama_batch_get_one(SegmentAllocator allocator, MemorySegment segment, int size) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_batch_get_one(allocator, segment, size);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_batch_get_one(allocator, segment, size);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  static int llama_decode(MemorySegment context, MemorySegment batch) {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_decode(context, batch);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_decode(context, batch);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }

  /* Free Memory */
  static void llama_batch_free(LlamaBatch batch) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_batch_free(batch.segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_batch_free(batch.segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static void llama_free(LlamaContext context) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_free(context.segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_free(context.segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static void llama_sampler_free(LlamaSampler sampler) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_sampler_free(sampler.segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_sampler_free(sampler.segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  static void llama_model_free(LlamaModel model) {
    switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_model_free(model.segment);
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_model_free(model.segment);
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    }
  }

  /* Utils */

  public static boolean llama_supports_gpu_offload() {
    return switch (runtime) {
      case MACOSX_AARCH_64 -> io.gravitee.llama.cpp.macosx.aarch64.llama_h.llama_supports_gpu_offload();
      case LINUX_X86_64 -> io.gravitee.llama.cpp.linux.x86_64.llama_h.llama_supports_gpu_offload();
      default -> throw new IllegalStateException("Unexpected value: " + runtime);
    };
  }
}
