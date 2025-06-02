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
public final class LlamaContext extends MemorySegmentAware implements Freeable {

  private final int nCtx;

  public LlamaContext(LlamaModel model, LlamaContextParams params) {
    super(llama_init_from_model(model.segment, params.segment));
    this.nCtx = params.nCtx();
  }

  public int nCtx() {
    return nCtx;
  }

  public int nCtxUsedCells() {
    return llama_get_kv_cache_used_cells(this.segment);
  }

  public void kvCacheClear() {
    llama_kv_cache_clear(this.segment);
  }

  @Override
  public void free() {
    llama_free(this);
  }
}
