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
 * The subset of {@code ggml_type} values usable for the KV cache data type
 * ({@link LlamaContextParams#typeK} / {@link LlamaContextParams#typeV}).
 *
 * <p>The native {@code ggml_type} enum is <b>non-contiguous</b> (gaps where legacy
 * quantizations were removed), so each constant carries its explicit
 * {@link #nativeValue()} — do <b>not</b> rely on {@link #ordinal()} to map to the C enum.
 *
 * <p>Only the KV-suitable types are listed. The K-quant family (Q4_K, Q5_K, …) is
 * intentionally omitted: its 256-element block size does not divide typical attention
 * head dimensions, so it is not valid for the KV cache.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public enum GgmlType {
  F32(0),
  F16(1),
  Q4_0(2),
  Q4_1(3),
  Q5_0(6),
  Q5_1(7),
  Q8_0(8),
  IQ4_NL(20),
  BF16(30);

  private final int nativeValue;

  GgmlType(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  /** The integer value of the corresponding {@code ggml_type} in llama.cpp. */
  public int nativeValue() {
    return nativeValue;
  }

  /** {@code true} for quantized types (everything except F32/F16/BF16). */
  public boolean isQuantized() {
    return this != F32 && this != F16 && this != BF16;
  }

  /**
   * Resolves a CLI/string token (e.g. {@code "q8_0"}, {@code "f16"}) to its constant,
   * case-insensitively.
   *
   * @throws LlamaException if the token is not a known KV-suitable type
   */
  public static GgmlType fromString(String token) {
    try {
      return valueOf(token.trim().toUpperCase());
    } catch (IllegalArgumentException e) {
      throw new LlamaException("Unsupported KV cache type: " + token);
    }
  }

  /**
   * Resolves a native {@code ggml_type} integer to its enum constant.
   *
   * @throws LlamaException if the value is not a KV-suitable type known here
   */
  public static GgmlType fromNative(int nativeValue) {
    for (GgmlType type : values()) {
      if (type.nativeValue == nativeValue) {
        return type;
      }
    }
    throw new LlamaException(
      "Unknown/unsupported ggml_type for KV cache: " + nativeValue
    );
  }
}
