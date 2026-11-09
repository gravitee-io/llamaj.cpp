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

import static io.gravitee.llama.cpp.cxx.CxxParam.BATCH_EXT;
import static io.gravitee.llama.cpp.cxx.CxxParam.CONST_MODEL;
import static io.gravitee.llama.cpp.cxx.CxxParam.DECISION_ORDER;
import static io.gravitee.llama.cpp.cxx.CxxParam.INT32;
import static io.gravitee.llama.cpp.cxx.CxxParam.UINT32;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import io.gravitee.llama.cpp.cxx.CxxFunction;
import io.gravitee.llama.cpp.cxx.CxxFunctions;
import io.gravitee.llama.cpp.platform.PlatformResolver;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.util.ArrayList;
import java.util.List;

/**
 * MTP "nextn" self-speculation, EAGLE3 layer-input extraction, the decision order read by a
 * joint decision head (clef, vllm-decision2), and the label texts of a decision model.
 *
 * <p>MTP and EAGLE3 are part of the public C API ({@code llama.h}) since llama.cpp v1.0.0 and
 * go through the jextract bindings in {@link LlamaRuntime}. {@link #available()} and
 * {@link #eagle3Available()} still check that the loaded {@code libllama} exports them, so a
 * native library from an older release (e.g. via {@code LLAMA_CPP_LIB_PATH}) is refused up front.
 *
 * <p>The decision order and the decision labels stay in llama.cpp's <b>staging</b> header
 * ({@code src/llama-ext.h}), which is not inside an {@code extern "C"} block: they are exported
 * under <b>C++-mangled</b> names (Itanium ABI, identical on macOS-arm64 and Linux-x86_64) that
 * jextract cannot bind, so each is declared as a {@link CxxFunction} and resolved through
 * {@link io.gravitee.llama.cpp.cxx}. If upstream changes a signature the mangled symbol vanishes
 * and {@link #decisionAvailable()} / {@link #decisionLabelsAvailable()} return {@code false};
 * {@code LlamaExtSymbolsTest} locks the derived manglings.
 *
 * @author GraviteeSource Team
 */
public final class LlamaExt {

  private LlamaExt() {}

  // MTP (nextn) self-speculation group, C symbols of llama.h.
  static final List<String> MTP_GROUP = List.of(
    "llama_set_embeddings_nextn",
    "llama_get_embeddings_nextn_ith",
    "llama_get_ctx_other"
  );

  // EAGLE3 layer-input extraction group, C symbols of llama.h.
  static final List<String> EAGLE3_GROUP = List.of(
    "llama_set_embeddings_layer_inp",
    "llama_get_embeddings_layer_inp",
    "llama_get_embeddings_nextn",
    "llama_model_target_layer_ids",
    "llama_model_target_layer_ids_n"
  );

  // Decision staging symbols (C++-mangled): joint decision head order (clef) and labels.
  static final CxxFunction BATCH_EXT_SET_DECISION_ORDER = CxxFunction.of(
    "llama_batch_ext_set_decision_order",
    JAVA_BOOLEAN,
    BATCH_EXT,
    INT32,
    DECISION_ORDER
  );
  static final CxxFunction MODEL_DECISION_LABEL = CxxFunction.of(
    "llama_model_decision_label",
    ADDRESS,
    CONST_MODEL,
    UINT32
  );

  static final List<CxxFunction> DECISION_GROUP = List.of(
    BATCH_EXT_SET_DECISION_ORDER
  );

  static final List<CxxFunction> DECISION_LABELS_GROUP = List.of(
    MODEL_DECISION_LABEL
  );

  private static boolean exported(String symbol) {
    return SymbolLookup.loaderLookup().find(symbol).isPresent();
  }

  private static boolean allExported(List<String> group) {
    return group.stream().allMatch(LlamaExt::exported);
  }

  private static String report(List<String> group) {
    StringBuilder sb = new StringBuilder();
    for (String symbol : group) {
      if (!sb.isEmpty()) {
        sb.append('\n');
      }
      sb
        .append(String.format("%-32s: ", symbol))
        .append(exported(symbol) ? "RESOLVED" : "MISSING");
    }
    return sb.toString();
  }

  /* ---------------------------------- public API ---------------------------------- */

  /**
   * True if the loaded {@code libllama} exports every MTP (nextn) function. When false, the
   * loaded llama.cpp build predates the public nextn API: do not attempt MTP against it.
   */
  public static boolean available() {
    return allExported(MTP_GROUP);
  }

  /** Per-symbol resolution report for the MTP group. */
  public static String resolutionReport() {
    return report(MTP_GROUP);
  }

  /** True if the loaded {@code libllama} exports every EAGLE3 function. */
  public static boolean eagle3Available() {
    return allExported(EAGLE3_GROUP);
  }

  /** Per-symbol resolution report for the EAGLE3 group. */
  public static String eagle3ResolutionReport() {
    return report(EAGLE3_GROUP);
  }

  /** True if the decision-order staging symbol (clef) resolved against the loaded {@code libllama}. */
  public static boolean decisionAvailable() {
    return CxxFunctions.allResolve(DECISION_GROUP);
  }

  /** Per-symbol resolution report for the decision staging group. */
  public static String decisionResolutionReport() {
    return CxxFunctions.report(DECISION_GROUP);
  }

  /** True if the decision-label staging symbol resolved against the loaded {@code libllama}. */
  public static boolean decisionLabelsAvailable() {
    return CxxFunctions.allResolve(DECISION_LABELS_GROUP);
  }

  /** Per-symbol resolution report for the decision-label staging group. */
  public static String decisionLabelsResolutionReport() {
    return CxxFunctions.report(DECISION_LABELS_GROUP);
  }

  /**
   * The label text of each option of a decision model ({@code <arch>.decision.labels}), in option
   * order; empty if the model declares none.
   */
  public static List<String> decisionLabels(LlamaModel model) {
    var labels = new ArrayList<String>();
    for (int i = 0; ; i++) {
      MemorySegment ptr = (MemorySegment) CxxFunctions.call(
        MODEL_DECISION_LABEL,
        model.segment,
        i
      );
      if (ptr == null || ptr.equals(MemorySegment.NULL)) {
        return List.copyOf(labels);
      }
      labels.add(ptr.reinterpret(Long.MAX_VALUE).getString(0));
    }
  }

  /**
   * Tag entry {@code idx} of an extended batch with the {@code llama_decision_order} a joint
   * decision head reads ({@link DecisionOrder#nativeValue()}).
   *
   * @return false if the index is out of the batch
   */
  public static boolean setDecisionOrder(
    MemorySegment batchExt,
    int idx,
    DecisionOrder order
  ) {
    return (boolean) CxxFunctions.call(
      BATCH_EXT_SET_DECISION_ORDER,
      batchExt,
      idx,
      order.nativeValue()
    );
  }

  /**
   * Turn on extraction of a context's pre-final-norm "nextn" hidden state (the MTP seed / the
   * EAGLE3 encoder output). {@code masked=false} stores rows densely by token position.
   */
  public static void setEmbeddingsNextn(
    LlamaContext ctx,
    boolean value,
    boolean masked
  ) {
    LlamaRuntime.llama_set_embeddings_nextn(ctx.segment, value, masked);
  }

  /**
   * Read the nextn hidden state for output row {@code i} as {@code nEmbd} floats. On a draft
   * (MTP/EAGLE3) context this is the seed for chaining the next draft token.
   */
  public static float[] getEmbeddingsNextnIth(
    LlamaContext ctx,
    int i,
    int nEmbd
  ) {
    MemorySegment ptr = LlamaRuntime.llama_get_embeddings_nextn_ith(
      ctx.segment,
      i
    );
    if (ptr == null || ptr.equals(MemorySegment.NULL)) {
      throw new LlamaException("no nextn embeddings at row " + i);
    }
    return ptr.reinterpret((long) nEmbd * Float.BYTES).toArray(JAVA_FLOAT);
  }

  /** The context this one was linked to via {@code ctx_other} (target for an MTP/EAGLE3 draft). */
  public static MemorySegment getCtxOther(LlamaContext ctx) {
    return LlamaRuntime.llama_get_ctx_other(ctx.segment);
  }

  /** Enable/disable capture of a target layer's input hidden state (EAGLE3 feature source). */
  public static void setEmbeddingsLayerInp(
    LlamaContext ctx,
    int layer,
    boolean value
  ) {
    LlamaRuntime.llama_set_embeddings_layer_inp(ctx.segment, layer, value);
  }

  /**
   * Raw pointer to the captured input hidden states of {@code layer} for the last decode:
   * {@code [n_tokens, n_embd]} floats. Caller reinterprets with the right size.
   */
  public static MemorySegment getEmbeddingsLayerInp(
    LlamaContext ctx,
    int layer
  ) {
    MemorySegment ptr = LlamaRuntime.llama_get_embeddings_layer_inp(
      ctx.segment,
      layer
    );
    if (ptr == null || ptr.equals(MemorySegment.NULL)) {
      throw new LlamaException("layer " + layer + " input not extracted");
    }
    return ptr;
  }

  /**
   * Raw pointer to the full nextn (pre-norm) embeddings buffer of the last encode/decode, one
   * {@code n_embd} row per output. EAGLE3 reads the encoder's g_embd rows from here.
   */
  public static MemorySegment getEmbeddingsNextnAll(LlamaContext ctx) {
    MemorySegment ptr = LlamaRuntime.llama_get_embeddings_nextn(ctx.segment);
    if (ptr == null || ptr.equals(MemorySegment.NULL)) {
      throw new LlamaException("no nextn embeddings buffer");
    }
    return ptr;
  }

  /**
   * The target-layer indices an EAGLE3/DFlash draft model was trained to consume
   * (empty for non-draft models).
   */
  public static int[] targetLayerIds(LlamaModel model) {
    int n = LlamaRuntime.llama_model_target_layer_ids_n(model.segment);
    if (n <= 0) {
      return new int[0];
    }
    MemorySegment ptr = LlamaRuntime.llama_model_target_layer_ids(
      model.segment
    );
    return ptr.reinterpret((long) n * Integer.BYTES).toArray(JAVA_INT);
  }

  /* ---------------- jextract-struct helper (not a staging symbol) ---------------- */

  private static final String BASE_PKG =
    "io.gravitee.llama.cpp." + PlatformResolver.platform().getPackage() + ".";

  // Cached once as a MethodHandle so the JIT can inline the invocation (near-zero overhead in the
  // hot path); a per-call reflective lookup would otherwise dominate a 1-layer MTP decode.
  private static final java.lang.invoke.MethodHandle BATCH_EMBD_SETTER =
    resolveBatchEmbdSetter();

  private static java.lang.invoke.MethodHandle resolveBatchEmbdSetter() {
    try {
      Class<?> clazz = Class.forName(BASE_PKG + "llama_batch");
      return java.lang.invoke.MethodHandles.publicLookup().findStatic(
        clazz,
        "embd",
        java.lang.invoke.MethodType.methodType(
          void.class,
          MemorySegment.class,
          MemorySegment.class
        )
      );
    } catch (ReflectiveOperationException e) {
      return null;
    }
  }

  /**
   * Set a batch's {@code embd} field pointer without clearing its {@code token} field — producing
   * the dual token+embd batch MTP/EAGLE3 need (native {@code llama_batch_init} allocates only one
   * of the two arrays). Uses the jextract-generated {@code llama_batch} struct accessor
   * (resolved once, cached).
   */
  public static void setBatchEmbd(LlamaBatch batch, MemorySegment embd) {
    if (BATCH_EMBD_SETTER == null) {
      throw new LlamaException("llama_batch.embd accessor not resolvable");
    }
    try {
      BATCH_EMBD_SETTER.invokeExact(batch.segment, embd);
    } catch (Throwable t) {
      throw new LlamaException("failed to set llama_batch.embd", t);
    }
  }

  /**
   * Set a pointer field of a {@code llama_batch} struct by name ({@code token}, {@code pos},
   * {@code n_seq_id}, {@code seq_id}, {@code logits}, {@code embd}) — used to build the raw
   * embd-only encoder batches EAGLE3 needs (all other arrays NULL so the native side
   * auto-fills them).
   */
  public static void setBatchPointer(
    LlamaBatch batch,
    String field,
    MemorySegment value
  ) {
    LlamaRuntime.invoke(
      "llama_batch",
      field,
      new Class<?>[] { MemorySegment.class, MemorySegment.class },
      batch.segment,
      value
    );
  }

  /** Set a {@code llama_batch}'s {@code n_tokens} field directly (raw encoder batches). */
  public static void setBatchNTokens(LlamaBatch batch, int nTokens) {
    LlamaRuntime.invoke(
      "llama_batch",
      "n_tokens",
      new Class<?>[] { MemorySegment.class, int.class },
      batch.segment,
      nTokens
    );
  }
}
