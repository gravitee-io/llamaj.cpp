package io.gravitee.llama.cpp;

import java.lang.foreign.Arena;

import static io.gravitee.llama.cpp.llama_h_1.llama_model_default_params;


/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaModelParams extends MemorySegmentAware {

	public LlamaModelParams(Arena arena) {
        super(llama_model_params.ofAddress(llama_model_default_params(arena), arena));
    }

	public int nGpuLayers() {
		return llama_model_params.n_gpu_layers$get(segment);
	}

	public LlamaModelParams nGpuLayers(int layers) {
		 llama_model_params.n_gpu_layers$set(segment, layers);
		 return this;
	}
}