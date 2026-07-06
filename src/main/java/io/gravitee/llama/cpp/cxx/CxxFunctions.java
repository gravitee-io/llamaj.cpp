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

import io.gravitee.llama.cpp.LlamaException;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Resolution + invocation engine for {@link CxxFunction} declarations: looks each symbol up
 * by its derived mangled name via {@link SymbolLookup#loaderLookup()} (the native libraries must
 * already be {@code System.load}-ed), builds the downcall handle once, and dispatches calls.
 *
 * <p>A symbol that fails to resolve is remembered as absent — {@link #call} then throws a
 * {@link LlamaException} naming the function and its expected mangling (fail-fast, never a
 * silent wrong-ABI call), and {@link #report} marks it {@code MISSING} so a broken llama.cpp
 * version bump is diagnosable with a single {@code nm} diff.
 *
 * @author GraviteeSource Team
 */
public final class CxxFunctions {

  private CxxFunctions() {}

  private static final Linker LINKER = Linker.nativeLinker();
  private static final SymbolLookup LOOKUP = SymbolLookup.loaderLookup();

  // Lazily-resolved handle cache; Optional.empty() = looked up, absent in the loaded libs.
  private static final Map<CxxFunction, Optional<MethodHandle>> CACHE =
    new ConcurrentHashMap<>();

  private static Optional<MethodHandle> handle(CxxFunction fn) {
    return CACHE.computeIfAbsent(fn, f ->
      LOOKUP.find(f.symbol()).map(seg ->
        LINKER.downcallHandle(seg, f.descriptor())
      )
    );
  }

  /** Whether this function's mangled symbol resolves against the loaded native libraries. */
  public static boolean resolves(CxxFunction fn) {
    return handle(fn).isPresent();
  }

  /** Whether every function in the group resolves. */
  public static boolean allResolve(List<CxxFunction> group) {
    for (CxxFunction fn : group) {
      if (!resolves(fn)) {
        return false;
      }
    }
    return true;
  }

  /** Per-symbol resolution report for a group (MISSING entries include the expected mangling). */
  public static String report(List<CxxFunction> group) {
    StringBuilder sb = new StringBuilder();
    for (CxxFunction fn : group) {
      if (!sb.isEmpty()) {
        sb.append('\n');
      }
      sb
        .append(String.format("%-32s: ", fn.name()))
        .append(resolves(fn) ? "RESOLVED" : "MISSING (" + fn.symbol() + ")");
    }
    return sb.toString();
  }

  /**
   * Invokes the function with the given arguments (boxed; negligible next to a native decode).
   * Throws {@link LlamaException} if the symbol is absent or the call fails.
   */
  public static Object call(CxxFunction fn, Object... args) {
    MethodHandle h = handle(fn).orElseThrow(() ->
      new LlamaException(
        "staging symbol not present in the loaded native libraries: " +
          fn.name() +
          " (" +
          fn.symbol() +
          ")"
      )
    );
    try {
      return h.invokeWithArguments(args);
    } catch (LlamaException e) {
      throw e;
    } catch (Throwable t) {
      throw new LlamaException(fn.name() + " failed", t);
    }
  }
}
