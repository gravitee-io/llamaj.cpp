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

import static io.gravitee.llama.cpp.cxx.CxxParam.BOOL;
import static io.gravitee.llama.cpp.cxx.CxxParam.CONST_MODEL;
import static io.gravitee.llama.cpp.cxx.CxxParam.CTX;
import static io.gravitee.llama.cpp.cxx.CxxParam.INT32;
import static io.gravitee.llama.cpp.cxx.CxxParam.UINT32;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import io.gravitee.llama.cpp.cxx.CxxFunction;
import io.gravitee.llama.cpp.cxx.CxxFunctions;
import io.gravitee.llama.cpp.platform.PlatformResolver;
import java.lang.foreign.MemorySegment;
import java.util.List;

/**
 * Binding for llama.cpp's <b>staging</b> API ({@code src/llama-ext.h}): MTP "nextn"
 * self-speculation and EAGLE3 layer-input extraction.
 *
 * <p>The staging header is not inside an {@code extern "C"} block, so these functions are
 * exported from {@code libllama} under <b>C++-mangled</b> names (Itanium ABI — identical on
 * macOS-arm64 and Linux-x86_64) and jextract cannot bind them. Each function is declared as a
 * {@link CxxFunction} data record; derivation of the mangled symbol, resolution, and
 * dispatch live in {@link io.gravitee.llama.cpp.cxx}. Adding a staging function is one
 * declaration line.
 *
 * <p><b>Fragility:</b> the mangled name encodes the full signature. If upstream renames a
 * function or changes a parameter type, the symbol vanishes and the group's {@code available()}
 * returns {@code false} — fail-fast, never a silent wrong-ABI call.
 * {@code MtpEagle3SpeculativeTest} asserts resolution in CI, so a llama.cpp version bump that
 * breaks the ABI fails with a per-symbol report; {@code LlamaExtSymbolsTest} locks the derived
 * manglings against the observed exports.
 *
 * @author GraviteeSource Team
 */
public final class LlamaExt {

  private LlamaExt() {}

  // MTP (nextn) self-speculation group.
  static final CxxFunction SET_EMBEDDINGS_NEXTN = CxxFunction.of(
    "llama_set_embeddings_nextn",
    null,
    CTX,
    BOOL,
    BOOL
  );
  static final CxxFunction GET_EMBEDDINGS_NEXTN_ITH = CxxFunction.of(
    "llama_get_embeddings_nextn_ith",
    ADDRESS,
    CTX,
    INT32
  );
  static final CxxFunction GET_CTX_OTHER = CxxFunction.of(
    "llama_get_ctx_other",
    ADDRESS,
    CTX
  );

  // EAGLE3 layer-input extraction group.
  static final CxxFunction SET_EMBEDDINGS_LAYER_INP = CxxFunction.of(
    "llama_set_embeddings_layer_inp",
    null,
    CTX,
    UINT32,
    BOOL
  );
  static final CxxFunction GET_EMBEDDINGS_LAYER_INP = CxxFunction.of(
    "llama_get_embeddings_layer_inp",
    ADDRESS,
    CTX,
    UINT32
  );
  static final CxxFunction GET_EMBEDDINGS_NEXTN = CxxFunction.of(
    "llama_get_embeddings_nextn",
    ADDRESS,
    CTX
  );
  static final CxxFunction MODEL_TARGET_LAYER_IDS = CxxFunction.of(
    "llama_model_target_layer_ids",
    ADDRESS,
    CONST_MODEL
  );
  static final CxxFunction MODEL_TARGET_LAYER_IDS_N = CxxFunction.of(
    "llama_model_target_layer_ids_n",
    JAVA_INT,
    CONST_MODEL
  );

  static final List<CxxFunction> MTP_GROUP = List.of(
    SET_EMBEDDINGS_NEXTN,
    GET_EMBEDDINGS_NEXTN_ITH,
    GET_CTX_OTHER
  );
  static final List<CxxFunction> EAGLE3_GROUP = List.of(
    SET_EMBEDDINGS_LAYER_INP,
    GET_EMBEDDINGS_LAYER_INP,
    GET_EMBEDDINGS_NEXTN,
    MODEL_TARGET_LAYER_IDS,
    MODEL_TARGET_LAYER_IDS_N
  );

  /* ---------------------------------- public API ---------------------------------- */

  /**
   * True if all MTP (nextn) staging symbols resolved against the loaded {@code libllama}. When
   * false, the loaded llama.cpp build either predates the nextn API or changed its signatures
   * (which changes the mangled names) — do not attempt MTP against it.
   */
  public static boolean available() {
    return CxxFunctions.allResolve(MTP_GROUP);
  }

  /** Per-symbol resolution report for the MTP staging group. */
  public static String resolutionReport() {
    return CxxFunctions.report(MTP_GROUP);
  }

  /** True if all EAGLE3 staging symbols resolved against the loaded {@code libllama}. */
  public static boolean eagle3Available() {
    return CxxFunctions.allResolve(EAGLE3_GROUP);
  }

  /** Per-symbol resolution report for the EAGLE3 staging group. */
  public static String eagle3ResolutionReport() {
    return CxxFunctions.report(EAGLE3_GROUP);
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
    CxxFunctions.call(SET_EMBEDDINGS_NEXTN, ctx.segment, value, masked);
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
    MemorySegment ptr = (MemorySegment) CxxFunctions.call(
      GET_EMBEDDINGS_NEXTN_ITH,
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
    return (MemorySegment) CxxFunctions.call(GET_CTX_OTHER, ctx.segment);
  }

  /** Enable/disable capture of a target layer's input hidden state (EAGLE3 feature source). */
  public static void setEmbeddingsLayerInp(
    LlamaContext ctx,
    int layer,
    boolean value
  ) {
    CxxFunctions.call(SET_EMBEDDINGS_LAYER_INP, ctx.segment, layer, value);
  }

  /**
   * Raw pointer to the captured input hidden states of {@code layer} for the last decode —
   * {@code [n_tokens, n_embd]} floats. Caller reinterprets with the right size.
   */
  public static MemorySegment getEmbeddingsLayerInp(
    LlamaContext ctx,
    int layer
  ) {
    MemorySegment ptr = (MemorySegment) CxxFunctions.call(
      GET_EMBEDDINGS_LAYER_INP,
      ctx.segment,
      layer
    );
    if (ptr == null || ptr.equals(MemorySegment.NULL)) {
      throw new LlamaException("layer " + layer + " input not extracted");
    }
    return ptr;
  }

  /**
   * Raw pointer to the full nextn (pre-norm) embeddings buffer of the last encode/decode —
   * one {@code n_embd} row per output. EAGLE3 reads the encoder's g_embd rows from here.
   */
  public static MemorySegment getEmbeddingsNextnAll(LlamaContext ctx) {
    MemorySegment ptr = (MemorySegment) CxxFunctions.call(
      GET_EMBEDDINGS_NEXTN,
      ctx.segment
    );
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
    int n = (int) CxxFunctions.call(MODEL_TARGET_LAYER_IDS_N, model.segment);
    if (n <= 0) {
      return new int[0];
    }
    MemorySegment ptr = (MemorySegment) CxxFunctions.call(
      MODEL_TARGET_LAYER_IDS,
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
