package io.gravitee.llama.cpp;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.util.Arrays;

import static io.gravitee.llama.cpp.llama_h_1.llama_max_devices;
import static io.gravitee.llama.cpp.llama_h_1.llama_model_default_params;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;


/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaModelParams extends MemorySegmentAware {

    public LlamaModelParams(Arena arena) {
        super(llama_model_params.ofAddress(llama_model_default_params(arena), arena));
        this.tensorSplit(arena, default_tensor_split());
    }

    private static float[] default_tensor_split() {
        int maxDevices = (int) llama_max_devices();
        if(maxDevices > 0){
            float[] tensorSplit = new float[maxDevices];
            Arrays.fill(tensorSplit, 1f / maxDevices);
            return tensorSplit;
        }
        return new float[0];
    }

    public int nGpuLayers() {
        return llama_model_params.n_gpu_layers$get(segment);
    }

    public LlamaModelParams nGpuLayers(int layers) {
        llama_model_params.n_gpu_layers$set(segment, layers);
        return this;
    }

    public SplitMode splitMode() {
        return SplitMode.fromOrdinal(llama_model_params.split_mode$get(segment));
    }

    public LlamaModelParams splitMode(SplitMode mode) {
        llama_model_params.split_mode$set(segment, mode.ordinal());
        return this;
    }

    public int mainGpu() {
        return llama_model_params.main_gpu$get(segment);
    }

    public LlamaModelParams mainGpu(int mainGpu) {
        llama_model_params.main_gpu$set(segment, mainGpu);
        return this;
    }

    public float[] tensorSplit() {
        var memorySegment = llama_model_params.tensor_split$get(segment);
        float[] tensorSplit = new float[(int) llama_max_devices()];
        for (int i = 0; i < llama_max_devices(); i++) {
            tensorSplit[i] = memorySegment.getAtIndex(JAVA_FLOAT, i);
        }
        return tensorSplit;
    }

    public LlamaModelParams tensorSplit(SegmentAllocator allocator, float[] tensorSplit) {
        var tensorSplitSegment = allocator.allocateArray(JAVA_FLOAT, llama_max_devices());
        MemorySegment.copy(
                MemorySegment.ofArray(tensorSplit),
                0,
                tensorSplitSegment,
                0,
                tensorSplit.length * JAVA_FLOAT.byteSize()
        );

        llama_model_params.tensor_split$set(segment, tensorSplitSegment);
        return this;
    }

    public boolean vocabOnly() {
        return llama_model_params.vocab_only$get(segment);
    }

    public LlamaModelParams vocabOnly(boolean vocabOnly) {
        llama_model_params.vocab_only$set(segment, vocabOnly);
        return this;
    }

    public boolean useMmap() {
        return llama_model_params.use_mmap$get(segment);
    }

    public LlamaModelParams useMmap(boolean useMmap) {
        llama_model_params.use_mmap$set(segment, useMmap);
        return this;
    }

    public boolean useMlock() {
        return llama_model_params.use_mlock$get(segment);
    }

    public LlamaModelParams useMlock(boolean useMlock) {
        llama_model_params.use_mlock$set(segment, useMlock);
        return this;
    }

    public boolean checkTensors() {
        return llama_model_params.check_tensors$get(segment);
    }

    public LlamaModelParams checkTensors(boolean checkTensors) {
        llama_model_params.check_tensors$set(segment, checkTensors);
        return this;
    }
}