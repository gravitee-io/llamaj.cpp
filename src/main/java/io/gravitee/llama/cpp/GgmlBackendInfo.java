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
 * Represents a registered GGML backend (e.g., CPU, Metal, CUDA, RPC).
 * Each backend may expose one or more {@link GgmlDeviceInfo} devices.
 *
 * @param name  The backend name as reported by ggml (e.g., "CPU", "Metal", "RPC").
 * @param index The index in the backend registry (0-based).
 *
 * @author Remi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record GgmlBackendInfo(String name, long index) {
  @Override
  public String toString() {
    return "GgmlBackendInfo[name=%s, index=%d]".formatted(name, index);
  }
}
