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
public enum OperatingSystem {
  MAC_OS_X("macosx"),
  LINUX("linux"),
  WINDOWS("windows");

  private final String osName;

  OperatingSystem(String osName) {
    this.osName = osName;
  }

  public static OperatingSystem fromSystem() {
    String osName = System.getProperty("os.name").toLowerCase();
    if (osName.contains("mac")) {
      return MAC_OS_X;
    }
    if (osName.contains("linux")) {
      return LINUX;
    }
    if (osName.contains("win")) {
      return WINDOWS;
    }
    throw new IllegalArgumentException("Unsupported operating system: " + osName);
  }

  public String getOsName() {
    return osName;
  }
}
