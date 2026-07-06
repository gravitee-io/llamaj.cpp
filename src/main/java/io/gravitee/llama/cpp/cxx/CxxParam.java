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
package io.gravitee.llama.cpp.cxx;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.MemoryLayout;

/**
 * C++ parameter kinds under the Itanium ABI, pairing the name-mangling token with the FFM
 * layout a downcall uses for it. Covers the types appearing in llama.cpp's staging API; extend
 * with further kinds (tokens per the Itanium C++ ABI mangling grammar) as upstream adds
 * signatures.
 *
 * <p>Pointer-to-struct kinds encode {@code P<len><name>} ({@code PK...} when const-qualified);
 * builtin kinds are single letters ({@code b} bool, {@code i} int, {@code j} unsigned int, ...).
 *
 * @author GraviteeSource Team
 */
public enum CxxParam {
  /** {@code llama_context*} */
  CTX("P13llama_context", ADDRESS),
  /** {@code const llama_model*} */
  CONST_MODEL("PK11llama_model", ADDRESS),
  /** {@code bool} */
  BOOL("b", JAVA_BOOLEAN),
  /** {@code int32_t} */
  INT32("i", JAVA_INT),
  /** {@code uint32_t} */
  UINT32("j", JAVA_INT);

  private final String mangle;
  private final MemoryLayout layout;

  CxxParam(String mangle, MemoryLayout layout) {
    this.mangle = mangle;
    this.layout = layout;
  }

  /** The Itanium mangling token for this parameter kind. */
  public String mangle() {
    return mangle;
  }

  /** The FFM layout a downcall passes this parameter as. */
  public MemoryLayout layout() {
    return layout;
  }
}
