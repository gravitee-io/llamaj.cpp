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
package io.gravitee.llama.cpp.platform;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public enum Architecture {
  X86_64("x86_64"),
  AARCH64("aarch64");

  private static final String OS_ARCH = "os.arch";

  private static final String AMD_64 = "amd64";
  private static final String X86_64_VALUE = "x86_64";

  private static final String AARCH64_VALUE = "aarch64";
  private static final String ARM_64 = "arm64";

  private final String arch;

  Architecture(String arch) {
    this.arch = arch;
  }

  public static Architecture fromSystem() {
    var osArch = System.getProperty(OS_ARCH).toLowerCase();
    if (osArch.contains(X86_64_VALUE) || osArch.contains(AMD_64)) {
      return X86_64;
    }

    if (osArch.contains(AARCH64_VALUE) || osArch.contains(ARM_64)) {
      return AARCH64;
    }

    throw new IllegalArgumentException("Unsupported operating system architecture: " + osArch);
  }

  public String getArch() {
    return arch;
  }
}
