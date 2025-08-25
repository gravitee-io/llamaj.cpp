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

import static io.gravitee.llama.cpp.LlamaRuntime.*;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class LlamaMemory extends MemorySegmentAware {

  public LlamaMemory(LlamaContext context) {
    super(llama_get_memory(context.segment));
  }

  public int posMin() {
    return llama_memory_seq_pos_min(this.segment, 0);
  }

  public int posMax() {
    return llama_memory_seq_pos_max(this.segment, 0);
  }

  public void clear() {
    llama_memory_clear(this.segment, true);
  }
}
