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
 * Represents a GGML backend device (e.g., a specific GPU, CPU core group, or RPC endpoint).
 * Each device belongs to a {@link GgmlBackendInfo} backend.
 *
 * @param name        The device name as reported by ggml (e.g., "Metal", "CUDA0", "RPC[192.168.1.10:50052]").
 * @param description A human-readable description of the device.
 * @param index       The global device index (0-based, across all backends).
 * @param backendName The name of the backend this device belongs to.
 *
 * @author Remi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record GgmlDeviceInfo(
  String name,
  String description,
  long index,
  String backendName
) {
  @Override
  public String toString() {
    return "GgmlDeviceInfo[name=%s, description=%s, index=%d, backend=%s]".formatted(
      name,
      description,
      index,
      backendName
    );
  }
}
