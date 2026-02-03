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
import java.lang.foreign.ValueLayout;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaRuntime {

  private static final Class<MemorySegment> MEM_SEG_CLASS = MemorySegment.class;

  private static final String pkg = PlatformResolver.platform().getPackage();
  private static final String runtime = PlatformResolver.platform().runtime();
  private static final String basePackage =
    "io.gravitee.llama.cpp.%s.".formatted(pkg);

  private LlamaRuntime() {}

  /* load backends */
  public static void llama_backend_init() {
    llama_h("llama_backend_init", new Class<?>[] {});
  }

  public static void ggml_backend_load_all_from_path(Arena arena, String path) {
    llama_h(
      "ggml_backend_load_all_from_path",
      new Class<?>[] { MemorySegment.class },
      arena.allocateFrom(path)
    );
  }

  public static void llama_backend_free() {
    llama_h("llama_backend_free", new Class<?>[] {});
  }

  public static long ggml_backend_reg_count() {
    return llama_h("ggml_backend_reg_count", new Class<?>[] {});
  }

  /* logging */
  public static void llama_log_set(MemorySegment m1, MemorySegment m2) {
    llama_h(
      "llama_log_set",
      new Class<?>[] { MEM_SEG_CLASS, MEM_SEG_CLASS },
      m1,
      m2
    );
  }

  /* Model Parameters */
  public static MemorySegment llama_model_default_params(Arena arena) {
    return llama_h(
      "llama_model_default_params",
      new Class<?>[] { SegmentAllocator.class },
      arena
    );
  }

  public static long llama_max_devices() {
    return llama_h("llama_max_devices", new Class<?>[] {});
  }

  public static int n_gpu_layers(MemorySegment segment) {
    return llama_model_params(
      "n_gpu_layers",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void n_gpu_layers(MemorySegment segment, int layers) {
    llama_model_params(
      "n_gpu_layers",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      layers
    );
  }

  public static int split_mode(MemorySegment segment) {
    return llama_model_params(
      "split_mode",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void split_mode(MemorySegment segment, int ordinal) {
    llama_model_params(
      "split_mode",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      ordinal
    );
  }

  public static int main_gpu(MemorySegment segment) {
    return llama_model_params(
      "main_gpu",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void main_gpu(MemorySegment segment, int mainGpu) {
    llama_model_params(
      "main_gpu",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      mainGpu
    );
  }

  public static MemorySegment tensor_split(MemorySegment segment) {
    return llama_model_params(
      "tensor_split",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void tensor_split(
    MemorySegment segment,
    MemorySegment tensorSplit
  ) {
    llama_model_params(
      "tensor_split",
      new Class<?>[] { MEM_SEG_CLASS, MEM_SEG_CLASS },
      segment,
      tensorSplit
    );
  }

  public static boolean vocab_only(MemorySegment segment) {
    return llama_model_params(
      "vocab_only",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void vocab_only(MemorySegment segment, boolean vocabOnly) {
    llama_model_params(
      "vocab_only",
      new Class<?>[] { MEM_SEG_CLASS, boolean.class },
      segment,
      vocabOnly
    );
  }

  public static boolean use_mmap(MemorySegment segment) {
    return llama_model_params(
      "use_mmap",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void use_mmap(MemorySegment segment, boolean useMmap) {
    llama_model_params(
      "use_mmap",
      new Class<?>[] { MEM_SEG_CLASS, boolean.class },
      segment,
      useMmap
    );
  }

  public static boolean use_mlock(MemorySegment segment) {
    return llama_model_params(
      "use_mlock",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void use_mlock(MemorySegment segment, boolean useMlock) {
    llama_model_params(
      "use_mlock",
      new Class<?>[] { MEM_SEG_CLASS, boolean.class },
      segment,
      useMlock
    );
  }

  public static boolean check_tensors(MemorySegment segment) {
    return llama_model_params(
      "check_tensors",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void check_tensors(
    MemorySegment segment,
    boolean checkTensors
  ) {
    llama_model_params(
      "check_tensors",
      new Class<?>[] { MEM_SEG_CLASS, boolean.class },
      segment,
      checkTensors
    );
  }

  /* Model */
  public static MemorySegment llama_model_load_from_file(
    MemorySegment modelPath,
    MemorySegment path
  ) {
    return llama_h(
      "llama_model_load_from_file",
      new Class<?>[] { MEM_SEG_CLASS, MEM_SEG_CLASS },
      modelPath,
      path
    );
  }

  public static MemorySegment llama_adapter_lora_init(
    MemorySegment model,
    MemorySegment path
  ) {
    return llama_h(
      "llama_adapter_lora_init",
      new Class<?>[] { MEM_SEG_CLASS, MEM_SEG_CLASS },
      model,
      path
    );
  }

  /* Vocab */
  public static MemorySegment llama_model_get_vocab(MemorySegment model) {
    return llama_h(
      "llama_model_get_vocab",
      new Class<?>[] { MEM_SEG_CLASS },
      model
    );
  }

  public static boolean llama_vocab_is_eog(MemorySegment vocab, int tokenId) {
    return llama_h(
      "llama_vocab_is_eog",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      vocab,
      tokenId
    );
  }

  public static int llama_token_to_piece(
    MemorySegment vocab,
    int token,
    MemorySegment buf,
    int length,
    int lstrip,
    boolean special
  ) {
    Class<?>[] parameterTypes = {
      MEM_SEG_CLASS,
      int.class,
      MEM_SEG_CLASS,
      int.class,
      int.class,
      boolean.class,
    };
    return llama_h(
      "llama_token_to_piece",
      parameterTypes,
      vocab,
      token,
      buf,
      length,
      lstrip,
      special
    );
  }

  /* Sampler */

  public static MemorySegment llama_sampler_chain_init(MemorySegment sampler) {
    return llama_h(
      "llama_sampler_chain_init",
      new Class<?>[] { MEM_SEG_CLASS },
      sampler
    );
  }

  public static MemorySegment llama_sampler_chain_default_params(
    SegmentAllocator allocator
  ) {
    return llama_h(
      "llama_sampler_chain_default_params",
      new Class<?>[] { SegmentAllocator.class },
      allocator
    );
  }

  public static int llama_sampler_sample(
    MemorySegment sampler,
    MemorySegment context,
    int idx
  ) {
    return llama_h(
      "llama_sampler_sample",
      new Class<?>[] { MEM_SEG_CLASS, MEM_SEG_CLASS, int.class },
      sampler,
      context,
      idx
    );
  }

  public static void llama_sampler_chain_add(
    MemorySegment sampler,
    MemorySegment config
  ) {
    llama_h(
      "llama_sampler_chain_add",
      new Class<?>[] { MEM_SEG_CLASS, MEM_SEG_CLASS },
      sampler,
      config
    );
  }

  public static MemorySegment llama_sampler_init_temp(float temperature) {
    return llama_h(
      "llama_sampler_init_temp",
      new Class<?>[] { float.class },
      temperature
    );
  }

  public static MemorySegment llama_sampler_init_greedy() {
    return llama_h("llama_sampler_init_greedy", new Class<?>[] {});
  }

  public static MemorySegment llama_sampler_init_top_k(int topK) {
    return llama_h(
      "llama_sampler_init_top_k",
      new Class<?>[] { int.class },
      topK
    );
  }

  public static MemorySegment llama_sampler_init_top_p(
    float topP,
    long minKeep
  ) {
    return llama_h(
      "llama_sampler_init_top_p",
      new Class<?>[] { float.class, long.class },
      topP,
      minKeep
    );
  }

  public static MemorySegment llama_sampler_init_min_p(
    float minP,
    long minKeep
  ) {
    return llama_h(
      "llama_sampler_init_min_p",
      new Class<?>[] { float.class, long.class },
      minP,
      minKeep
    );
  }

  public static MemorySegment llama_sampler_init_mirostat_v2(
    int seed,
    float tau,
    float eta
  ) {
    return llama_h(
      "llama_sampler_init_mirostat_v2",
      new Class<?>[] { int.class, float.class, float.class },
      seed,
      tau,
      eta
    );
  }

  public static MemorySegment llama_sampler_init_grammar(
    MemorySegment vocab,
    MemorySegment grammar,
    MemorySegment root
  ) {
    return llama_h(
      "llama_sampler_init_grammar",
      new Class<?>[] { MEM_SEG_CLASS, MEM_SEG_CLASS, MEM_SEG_CLASS },
      vocab,
      grammar,
      root
    );
  }

  public static MemorySegment llama_sampler_init_penalties(
    int penaltyLastN,
    float penaltyRepeat,
    float penaltyFreq,
    float penaltyPresent
  ) {
    return llama_h(
      "llama_sampler_init_penalties",
      new Class<?>[] { int.class, float.class, float.class, float.class },
      penaltyLastN,
      penaltyRepeat,
      penaltyFreq,
      penaltyPresent
    );
  }

  public static MemorySegment llama_sampler_init_dist(int seed) {
    return llama_h(
      "llama_sampler_init_dist",
      new Class<?>[] { int.class },
      seed
    );
  }

  /* Context Params */
  public static MemorySegment llama_context_default_params(
    SegmentAllocator allocator
  ) {
    return llama_h(
      "llama_context_default_params",
      new Class<?>[] { SegmentAllocator.class },
      allocator
    );
  }

  public static int n_ctx(MemorySegment contextParams) {
    return llama_context_params(
      "n_ctx",
      new Class<?>[] { MEM_SEG_CLASS },
      contextParams
    );
  }

  public static void n_ctx(MemorySegment segment, int nCtx) {
    llama_context_params(
      "n_ctx",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      nCtx
    );
  }

  public static int n_batch(MemorySegment segment) {
    return llama_context_params(
      "n_batch",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void n_batch(MemorySegment segment, int nBatch) {
    llama_context_params(
      "n_batch",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      nBatch
    );
  }

  public static int n_ubatch(MemorySegment segment) {
    return llama_context_params(
      "n_ubatch",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void n_ubatch(MemorySegment segment, int nUBatch) {
    llama_context_params(
      "n_ubatch",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      nUBatch
    );
  }

  public static int n_seq_max(MemorySegment segment) {
    return llama_context_params(
      "n_seq_max",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void n_seq_max(MemorySegment segment, int nSeqMax) {
    llama_context_params(
      "n_seq_max",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      nSeqMax
    );
  }

  public static int n_threads(MemorySegment segment) {
    return llama_context_params(
      "n_threads",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void n_threads(MemorySegment segment, int nThreads) {
    llama_context_params(
      "n_threads",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      nThreads
    );
  }

  public static int n_threads_batch(MemorySegment segment) {
    return llama_context_params(
      "n_threads_batch",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void n_threads_batch(MemorySegment segment, int nThreadBatch) {
    llama_context_params(
      "n_threads_batch",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      nThreadBatch
    );
  }

  public static int pooling_type(MemorySegment segment) {
    return llama_context_params(
      "pooling_type",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void pooling_type(MemorySegment segment, int ordinal) {
    llama_context_params(
      "pooling_type",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      ordinal
    );
  }

  public static int attention_type(MemorySegment segment) {
    return llama_context_params(
      "attention_type",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void attention_type(MemorySegment segment, int ordinal) {
    llama_context_params(
      "attention_type",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      ordinal
    );
  }

  public static boolean embeddings(MemorySegment segment) {
    return llama_context_params(
      "embeddings",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void embeddings(MemorySegment segment, boolean embeddings) {
    llama_context_params(
      "embeddings",
      new Class<?>[] { MEM_SEG_CLASS, boolean.class },
      segment,
      embeddings
    );
  }

  public static boolean offload_kqv(MemorySegment segment) {
    return llama_context_params(
      "offload_kqv",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void offload_kqv(MemorySegment segment, boolean offloadKQV) {
    llama_context_params(
      "offload_kqv",
      new Class<?>[] { MEM_SEG_CLASS, boolean.class },
      segment,
      offloadKQV
    );
  }

  public static int flash_attn_type(MemorySegment segment) {
    return llama_context_params(
      "flash_attn_type",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void flash_attn_type(MemorySegment segment, int flashAttnType) {
    llama_context_params(
      "flash_attn_type",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      segment,
      flashAttnType
    );
  }

  public static boolean no_perf(MemorySegment segment) {
    return llama_context_params(
      "no_perf",
      new Class<?>[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void no_perf(MemorySegment segment, boolean noPerf) {
    llama_context_params(
      "no_perf",
      new Class<?>[] { MEM_SEG_CLASS, boolean.class },
      segment,
      noPerf
    );
  }

  /* Context */
  public static MemorySegment llama_init_from_model(
    MemorySegment model,
    MemorySegment params
  ) {
    return llama_h(
      "llama_init_from_model",
      new Class<?>[] { MEM_SEG_CLASS, MEM_SEG_CLASS },
      model,
      params
    );
  }

  /* Memory */

  public static MemorySegment llama_get_memory(MemorySegment ctx) {
    return llama_h("llama_get_memory", new Class<?>[] { MEM_SEG_CLASS }, ctx);
  }

  public static int llama_memory_seq_pos_max(MemorySegment memory, int seq_id) {
    return llama_h(
      "llama_memory_seq_pos_max",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      memory,
      seq_id
    );
  }

  public static int llama_memory_seq_pos_min(MemorySegment memory, int seq_id) {
    return llama_h(
      "llama_memory_seq_pos_min",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      memory,
      seq_id
    );
  }

  public static void llama_memory_clear(MemorySegment memory, boolean data) {
    llama_h(
      "llama_memory_clear",
      new Class<?>[] { MEM_SEG_CLASS, boolean.class },
      memory,
      data
    );
  }

  public static boolean llama_memory_seq_rm(
    MemorySegment memory,
    int seqId,
    int p0,
    int p1
  ) {
    return llama_h(
      "llama_memory_seq_rm",
      new Class<?>[] { MEM_SEG_CLASS, int.class, int.class, int.class },
      memory,
      seqId,
      p0,
      p1
    );
  }

  public static void llama_memory_seq_cp(
    MemorySegment memory,
    int seqIdSrc,
    int seqIdDst,
    int p0,
    int p1
  ) {
    llama_h(
      "llama_memory_seq_cp",
      new Class<?>[] {
        MEM_SEG_CLASS,
        int.class,
        int.class,
        int.class,
        int.class,
      },
      memory,
      seqIdSrc,
      seqIdDst,
      p0,
      p1
    );
  }

  public static void llama_memory_seq_keep(MemorySegment memory, int seqId) {
    llama_h(
      "llama_memory_seq_keep",
      new Class<?>[] { MEM_SEG_CLASS, int.class },
      memory,
      seqId
    );
  }

  public static void llama_memory_seq_add(
    MemorySegment memory,
    int seqId,
    int p0,
    int p1,
    int delta
  ) {
    llama_h(
      "llama_memory_seq_add",
      new Class<?>[] {
        MEM_SEG_CLASS,
        int.class,
        int.class,
        int.class,
        int.class,
      },
      memory,
      seqId,
      p0,
      p1,
      delta
    );
  }

  public static void llama_memory_seq_div(
    MemorySegment memory,
    int seqId,
    int p0,
    int p1,
    int d
  ) {
    llama_h(
      "llama_memory_seq_div",
      new Class<?>[] {
        MEM_SEG_CLASS,
        int.class,
        int.class,
        int.class,
        int.class,
      },
      memory,
      seqId,
      p0,
      p1,
      d
    );
  }

  /* Chat Template */
  public static MemorySegment llama_model_chat_template(
    MemorySegment model,
    MemorySegment name
  ) {
    return llama_h(
      "llama_model_chat_template",
      new Class<?>[] { MEM_SEG_CLASS, MEM_SEG_CLASS },
      model,
      name
    );
  }

  public static int llama_chat_apply_template(
    MemorySegment tmpl,
    MemorySegment chat,
    long n_msg,
    boolean add_ass,
    MemorySegment buf,
    int length
  ) {
    return llama_h(
      "llama_chat_apply_template",
      new Class<?>[] {
        MEM_SEG_CLASS,
        MEM_SEG_CLASS,
        long.class,
        boolean.class,
        MEM_SEG_CLASS,
        int.class,
      },
      tmpl,
      chat,
      n_msg,
      add_ass,
      buf,
      length
    );
  }

  /* Chat Message */
  public static MemorySegment llama_chat_message_allocate(
    SegmentAllocator allocator
  ) {
    return llama_chat_message(
      "allocate",
      new Class[] { SegmentAllocator.class },
      allocator
    );
  }

  public static MemorySegment llama_chat_message_allocateArray(
    long size,
    SegmentAllocator allocator
  ) {
    return llama_chat_message(
      "allocateArray",
      new Class[] { long.class, SegmentAllocator.class },
      size,
      allocator
    );
  }

  public static MemorySegment llama_chat_message_role(MemorySegment segment) {
    return llama_chat_message("role", new Class[] { MEM_SEG_CLASS }, segment);
  }

  public static void llama_chat_message_role(
    MemorySegment message,
    MemorySegment role
  ) {
    llama_chat_message(
      "role",
      new Class[] { MEM_SEG_CLASS, MEM_SEG_CLASS },
      message,
      role
    );
  }

  public static MemorySegment llama_chat_message_content(
    MemorySegment segment
  ) {
    return llama_chat_message(
      "content",
      new Class[] { MEM_SEG_CLASS },
      segment
    );
  }

  public static void llama_chat_message_content(
    MemorySegment message,
    MemorySegment content
  ) {
    llama_chat_message(
      "content",
      new Class[] { MEM_SEG_CLASS, MEM_SEG_CLASS },
      message,
      content
    );
  }

  public static long llama_chat_message_sizeof() {
    return llama_chat_message("sizeof", new Class[] {});
  }

  /* Tokenizer */
  public static int llama_tokenize(
    MemorySegment vocab,
    MemorySegment text,
    int text_len,
    MemorySegment tokens,
    int n_tokens_max,
    boolean add_special,
    boolean parse_special
  ) {
    return llama_h(
      "llama_tokenize",
      new Class[] {
        MEM_SEG_CLASS,
        MEM_SEG_CLASS,
        int.class,
        MEM_SEG_CLASS,
        int.class,
        boolean.class,
        boolean.class,
      },
      vocab,
      text,
      text_len,
      tokens,
      n_tokens_max,
      add_special,
      parse_special
    );
  }

  /* Batch */
  public static MemorySegment llama_batch_get_one(
    SegmentAllocator allocator,
    MemorySegment segment,
    int size
  ) {
    return llama_h(
      "llama_batch_get_one",
      new Class[] { SegmentAllocator.class, MEM_SEG_CLASS, int.class },
      allocator,
      segment,
      size
    );
  }

  public static MemorySegment llama_batch_init(
    SegmentAllocator allocator,
    int nTokens,
    int embd,
    int nSeqMax
  ) {
    return llama_h(
      "llama_batch_init",
      new Class[] { SegmentAllocator.class, int.class, int.class, int.class },
      allocator,
      nTokens,
      embd,
      nSeqMax
    );
  }

  public static void llama_batch_add(
    MemorySegment batch,
    int token,
    int pos,
    java.util.List<Integer> seqIds,
    boolean logits
  ) {
    // Get batch struct fields
    MemorySegment tokens = invoke(
      "llama_batch",
      "token",
      new Class[] { MEM_SEG_CLASS },
      batch
    );
    MemorySegment positions = invoke(
      "llama_batch",
      "pos",
      new Class[] { MEM_SEG_CLASS },
      batch
    );
    MemorySegment nSeqId = invoke(
      "llama_batch",
      "n_seq_id",
      new Class[] { MEM_SEG_CLASS },
      batch
    );
    MemorySegment seqIdPtr = invoke(
      "llama_batch",
      "seq_id",
      new Class[] { MEM_SEG_CLASS },
      batch
    );
    MemorySegment logitsPtr = invoke(
      "llama_batch",
      "logits",
      new Class[] { MEM_SEG_CLASS },
      batch
    );
    int nTokens = llama_batch_n_tokens(batch);

    // Set token
    tokens.setAtIndex(ValueLayout.JAVA_INT, nTokens, token);
    // Set position
    positions.setAtIndex(ValueLayout.JAVA_INT, nTokens, pos);
    // Set number of sequence IDs
    nSeqId.setAtIndex(ValueLayout.JAVA_INT, nTokens, seqIds.size());

    // Set sequence IDs
    // Get the pointer to the array of seq_ids for this token position
    // The pointer needs to be reinterpreted with the correct size (seqIds.size() * sizeof(int))
    MemorySegment seqIdArray = seqIdPtr
      .getAtIndex(ValueLayout.ADDRESS, nTokens)
      .reinterpret(seqIds.size() * Integer.BYTES);
    for (int i = 0; i < seqIds.size(); i++) {
      seqIdArray.setAtIndex(ValueLayout.JAVA_INT, i, seqIds.get(i));
    }

    // Set logits flag
    logitsPtr.setAtIndex(
      ValueLayout.JAVA_BYTE,
      nTokens,
      logits ? (byte) 1 : (byte) 0
    );

    // Increment n_tokens
    invoke(
      "llama_batch",
      "n_tokens",
      new Class[] { MEM_SEG_CLASS, int.class },
      batch,
      nTokens + 1
    );
  }

  public static void llama_batch_clear(MemorySegment batch) {
    invoke(
      "llama_batch",
      "n_tokens",
      new Class[] { MEM_SEG_CLASS, int.class },
      batch,
      0
    );
  }

  public static int llama_batch_n_tokens(MemorySegment batch) {
    return invoke(
      "llama_batch",
      "n_tokens",
      new Class[] { MEM_SEG_CLASS },
      batch
    );
  }

  public static int llama_decode(MemorySegment context, MemorySegment batch) {
    return llama_h(
      "llama_decode",
      new Class[] { MEM_SEG_CLASS, MEM_SEG_CLASS },
      context,
      batch
    );
  }

  /* Free Memory */
  public static void llama_batch_free(LlamaBatch batch) {
    llama_h("llama_batch_free", new Class[] { MEM_SEG_CLASS }, batch.segment);
  }

  public static void llama_free(LlamaContext context) {
    llama_h("llama_free", new Class[] { MEM_SEG_CLASS }, context.segment);
  }

  public static void llama_sampler_free(LlamaSampler sampler) {
    llama_h(
      "llama_sampler_free",
      new Class[] { MEM_SEG_CLASS },
      sampler.segment
    );
  }

  public static void llama_model_free(LlamaModel model) {
    llama_h("llama_model_free", new Class[] { MEM_SEG_CLASS }, model.segment);
  }

  public static void llama_adapter_lora_free(LlamaLoraAdapter adapter) {
    llama_h(
      "llama_adapter_lora_free",
      new Class[] { MEM_SEG_CLASS },
      adapter.segment
    );
  }

  /* Utils */

  public static boolean llama_supports_gpu_offload() {
    return llama_h("llama_supports_gpu_offload", new Class[] {});
  }

  /* Performance */

  public static MemorySegment llama_perf_context(
    SegmentAllocator allocator,
    MemorySegment ctx
  ) {
    return llama_h(
      "llama_perf_context",
      new Class[] { SegmentAllocator.class, MEM_SEG_CLASS },
      allocator,
      ctx
    );
  }

  public static MemorySegment llama_perf_sampler(
    SegmentAllocator allocator,
    MemorySegment sampler
  ) {
    return llama_h(
      "llama_perf_sampler",
      new Class[] { SegmentAllocator.class, MEM_SEG_CLASS },
      allocator,
      sampler
    );
  }

  public static double llama_perf_context_t_start_ms(MemorySegment perfData) {
    return invoke(
      "llama_perf_context_data",
      "t_start_ms",
      new Class[] { MEM_SEG_CLASS },
      perfData
    );
  }

  public static double llama_perf_context_t_load_ms(MemorySegment perfData) {
    return invoke(
      "llama_perf_context_data",
      "t_load_ms",
      new Class[] { MEM_SEG_CLASS },
      perfData
    );
  }

  public static double llama_perf_context_t_p_eval_ms(MemorySegment perfData) {
    return invoke(
      "llama_perf_context_data",
      "t_p_eval_ms",
      new Class[] { MEM_SEG_CLASS },
      perfData
    );
  }

  public static double llama_perf_context_t_eval_ms(MemorySegment perfData) {
    return invoke(
      "llama_perf_context_data",
      "t_eval_ms",
      new Class[] { MEM_SEG_CLASS },
      perfData
    );
  }

  public static int llama_perf_context_n_p_eval(MemorySegment perfData) {
    return invoke(
      "llama_perf_context_data",
      "n_p_eval",
      new Class[] { MEM_SEG_CLASS },
      perfData
    );
  }

  public static int llama_perf_context_n_eval(MemorySegment perfData) {
    return invoke(
      "llama_perf_context_data",
      "n_eval",
      new Class[] { MEM_SEG_CLASS },
      perfData
    );
  }

  public static int llama_perf_context_n_reused(MemorySegment perfData) {
    return invoke(
      "llama_perf_context_data",
      "n_reused",
      new Class[] { MEM_SEG_CLASS },
      perfData
    );
  }

  public static double llama_perf_sampler_t_sample_ms(MemorySegment perfData) {
    return invoke(
      "llama_perf_sampler_data",
      "t_sample_ms",
      new Class[] { MEM_SEG_CLASS },
      perfData
    );
  }

  public static int llama_perf_sampler_n_sample(MemorySegment perfData) {
    return invoke(
      "llama_perf_sampler_data",
      "n_sample",
      new Class[] { MEM_SEG_CLASS },
      perfData
    );
  }

  /**
   * Dynamically invokes a static method on the correct platform-specific llama_h class.
   *
   * @param methodName      The name of the method to invoke.
   * @param parameterTypes  An array of Class objects representing the parameter types of the method.
   * @param args            The arguments to pass to the method.
   * @return The result of the method invocation.
   * @throws IllegalStateException    If the runtime is unknown or if reflection fails.
   */
  public static <T> T llama_h(
    String methodName,
    Class<?>[] parameterTypes,
    Object... args
  ) {
    return invoke("llama_h", methodName, parameterTypes, args);
  }

  /**
   * Dynamically invokes a static method on the correct platform-specific llama_model_params class.
   *
   * @param methodName      The name of the method to invoke.
   * @param parameterTypes  An array of Class objects representing the parameter types of the method.
   * @param args            The arguments to pass to the method.
   * @return The result of the method invocation.
   * @throws IllegalStateException    If the runtime is unknown or if reflection fails.
   */
  public static <T> T llama_model_params(
    String methodName,
    Class<?>[] parameterTypes,
    Object... args
  ) {
    return invoke("llama_model_params", methodName, parameterTypes, args);
  }

  /**
   * Dynamically invokes a static method on the correct platform-specific llama_context_params class.
   *
   * @param methodName      The name of the method to invoke.
   * @param parameterTypes  An array of Class objects representing the parameter types of the method.
   * @param args            The arguments to pass to the method.
   * @return The result of the method invocation.
   * @throws IllegalStateException    If the runtime is unknown or if reflection fails.
   */
  public static <T> T llama_context_params(
    String methodName,
    Class<?>[] parameterTypes,
    Object... args
  ) {
    return invoke("llama_context_params", methodName, parameterTypes, args);
  }

  /**
   * Dynamically invokes a static method on the correct platform-specific llama_chat_message class.
   *
   * @param methodName      The name of the method to invoke.
   * @param parameterTypes  An array of Class objects representing the parameter types of the method.
   * @param args            The arguments to pass to the method.
   * @return The result of the method invocation.
   * @throws IllegalStateException    If the runtime is unknown or if reflection fails.
   */
  public static <T> T llama_chat_message(
    String methodName,
    Class<?>[] parameterTypes,
    Object... args
  ) {
    return invoke("llama_chat_message", methodName, parameterTypes, args);
  }

  /**
   * Dynamically invokes a static method on the correct platform-specific class.
   *
   * @param classNameSuffix The suffix of the class name (e.g., "llama_h", "llama_model_params").
   * @param methodName      The name of the method to invoke.
   * @param parameterTypes  An array of Class objects representing the parameter types of the method.
   * @param args            The arguments to pass to the method.
   * @return The result of the method invocation.
   * @throws IllegalStateException    If the runtime is unknown or if reflection fails.
   */
  public static <T> T invoke(
    String classNameSuffix,
    String methodName,
    Class<?>[] parameterTypes,
    Object... args
  ) {
    try {
      String fullClassName = basePackage + classNameSuffix;
      Class<?> targetClass = Class.forName(fullClassName);
      Method method = targetClass.getMethod(methodName, parameterTypes);
      return (T) method.invoke(null, args); // Invoke static method, so obj is null
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(
        "Class not found for runtime " + runtime + ": " + e.getMessage(),
        e
      );
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(
        "Method not found for runtime " + runtime + ": " + e.getMessage(),
        e
      );
    } catch (InvocationTargetException e) {
      if (e.getTargetException() instanceof RuntimeException) {
        throw (RuntimeException) e.getTargetException();
      }
      throw new IllegalStateException(
        "Error invoking method for runtime " + runtime + ": " + e.getMessage(),
        e
      );
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(
        "Illegal access to method for runtime " +
          runtime +
          ": " +
          e.getMessage(),
        e
      );
    }
  }

  /**
   * Instantiates a MemorySegment for a functional interface using reflection.
   *
   * @param callbackInterfaceName The simple name of the jextract-generated functional interface (e.g., "ggml_log_callback").
   * @param callbackMethodName The name of the static factory method (e.g., "allocate").
   * @param targetObject     The object on which the actual callback method will be invoked.
   * @param targetMethodName The name of the method on the targetObject to be invoked by the native callback.
   * @param arena            The Arena to allocate the MemorySegment in.
   * @return A MemorySegment representing the native function pointer for the callback.
   * @throws Exception if any reflective operation fails.
   */
  public static MemorySegment instantiateCallbackSegment(
    String callbackInterfaceName,
    String callbackMethodName,
    Object targetObject,
    String targetMethodName,
    Arena arena
  ) throws Exception {
    String fullInterfaceName = basePackage + callbackInterfaceName;

    // --- 1. Load the interface class ---
    Class<?> callbackInterface;
    try {
      callbackInterface = Class.forName(fullInterfaceName);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(
        "jextract-generated callback interface not found: " +
          fullInterfaceName +
          "\nVerify that jextract ran successfully and the classpath includes generated classes.",
        e
      );
    }

    // --- 2. Validate that the interface has a nested Function interface ---
    Class<?> functionInterface = null;
    for (Class<?> inner : callbackInterface.getDeclaredClasses()) {
      if ("Function".equals(inner.getSimpleName())) {
        functionInterface = inner;
        break;
      }
    }

    if (functionInterface == null) {
      throw new IllegalStateException(
        "Expected inner interface 'Function' not found in " +
          fullInterfaceName +
          "\nThis suggests jextract generation has changed. Expected structure: " +
          "class ggml_log_callback { public interface Function { void apply(...); } }"
      );
    }

    // --- 3. Create dynamic proxy implementing the Function interface ---
    Object proxyCallbackInstance = Proxy.newProxyInstance(
      callbackInterface.getClassLoader(),
      new Class<?>[] { functionInterface },
      (proxy, method, args) -> {
        try {
          Method targetMethod = targetObject
            .getClass()
            .getMethod(targetMethodName, method.getParameterTypes());
          return targetMethod.invoke(targetObject, args);
        } catch (NoSuchMethodException e) {
          throw new UnsupportedOperationException(
            "Callback method '" +
              method.getName() +
              "' not found on target object '" +
              targetObject.getClass().getName() +
              "' with matching signature.",
            e
          );
        } catch (InvocationTargetException e) {
          throw e.getCause();
        }
      }
    );

    // --- 4. Find the static allocate method: (Function, Arena) -> MemorySegment ---
    Method allocateMethod;
    try {
      allocateMethod = callbackInterface.getMethod(
        callbackMethodName,
        functionInterface,
        Arena.class
      );
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(
        "Static method '" +
          callbackMethodName +
          "' with signature (" +
          functionInterface.getName() +
          ", Arena) not found in " +
          fullInterfaceName +
          "\nThis suggests jextract generation has changed. Expected: " +
          "public static MemorySegment allocate(Function, Arena)",
        e
      );
    }

    // --- 5. Invoke allocate(null, proxy, arena) ---
    try {
      return (MemorySegment) allocateMethod.invoke(
        null,
        proxyCallbackInstance,
        arena
      );
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(
        "Cannot access " +
          fullInterfaceName +
          "." +
          callbackMethodName +
          ". Ensure classes are in the same module or module is exported.",
        e
      );
    } catch (InvocationTargetException e) {
      throw new RuntimeException("Failed to create upcall stub", e.getCause());
    }
  }
}
