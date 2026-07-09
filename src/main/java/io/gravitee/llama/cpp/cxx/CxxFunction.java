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

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.util.List;

/**
 * A C++-exported free function as data: C name, return layout ({@code null} = void), and
 * {@link CxxParam} parameter kinds. Everything a binding needs is derived from the
 * signature: the Itanium-mangled symbol name ({@code _Z<len><name><params>} — return types are
 * not encoded for free functions) and the FFM {@link FunctionDescriptor}.
 *
 * <p>Declared as static constants and resolved/invoked through
 * {@link CxxFunctions} — adding a binding is one declaration line.
 *
 * @param name   The unmangled C++ function name
 * @param ret    Return layout, or {@code null} for {@code void}
 * @param params Parameter kinds, in order
 * @author GraviteeSource Team
 */
public record CxxFunction(
  String name,
  MemoryLayout ret,
  List<CxxParam> params
) {
  public static CxxFunction of(
    String name,
    MemoryLayout ret,
    CxxParam... params
  ) {
    return new CxxFunction(name, ret, List.of(params));
  }

  /** The Itanium-mangled symbol name derived from the signature. */
  public String symbol() {
    StringBuilder sb = new StringBuilder("_Z")
      .append(name.length())
      .append(name);
    for (CxxParam p : params) {
      sb.append(p.mangle());
    }
    return sb.toString();
  }

  /** The FFM downcall descriptor derived from the signature. */
  public FunctionDescriptor descriptor() {
    MemoryLayout[] args = params
      .stream()
      .map(CxxParam::layout)
      .toArray(MemoryLayout[]::new);
    return ret == null
      ? FunctionDescriptor.ofVoid(args)
      : FunctionDescriptor.of(ret, args);
  }
}
