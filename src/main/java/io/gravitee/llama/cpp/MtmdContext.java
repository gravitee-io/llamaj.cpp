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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

/**
 * Wrapper for the native `mtmd_context`.
 * This class handles the initialization and freeing of the native context.
 */
public class MtmdContext extends MemorySegmentAware implements Freeable {

  public MtmdContext(
    Arena arena,
    LlamaModel llamaModel,
    Path mmprojFilePath,
    MtmdContextParams mtmdContextParams
  ) {
    super(
      mtmd_init_from_file(
        getMmprojPath(arena, mmprojFilePath),
        llamaModel.segment,
        mtmdContextParams.segment
      )
    );
  }

  private static MemorySegment getMmprojPath(Arena arena, Path mmprojFilePath) {
    byte[] utf8 = mmprojFilePath.toString().getBytes(StandardCharsets.UTF_8);
    MemorySegment cString = arena.allocate(utf8.length + 1);

    cString.asSlice(0, utf8.length).copyFrom(MemorySegment.ofArray(utf8));

    cString.set(ValueLayout.JAVA_BYTE, utf8.length, (byte) 0);
    return cString;
  }

  public boolean supportsVision() {
    return mtmd_support_vision(segment);
  }

  public boolean supportsAudio() {
    return mtmd_support_audio(segment);
  }

  public int getAudioBitrate() {
    return mtmd_get_audio_bitrate(segment);
  }

  public int encodeChunk(MemorySegment mtmdInputChunkSegment) {
    return mtmd_encode_chunk(segment, mtmdInputChunkSegment);
  }

  public MemorySegment getOutputEmbd() {
    return mtmd_get_output_embd(segment);
  }

  /**
   * Evaluates all multimodal input chunks using the native helper.
   * This handles text, image, and audio chunks with proper M-RoPE positions,
   * non-causal attention, and batch splitting — matching the reference server exactly.
   *
   * @param arena Memory arena for allocations
   * @param llamaContext The llama context for decoding
   * @param chunks The tokenized input chunks
   * @param nPast Starting position in the KV cache
   * @param seqId Sequence ID
   * @param nBatch Maximum batch size
   * @param logitsLast Whether to compute logits for the last token
   * @return The updated nPast after all chunks are processed
   */
  public long evalChunks(
    Arena arena,
    LlamaContext llamaContext,
    MtmdInputChunks chunks,
    int nPast,
    int seqId,
    int nBatch,
    boolean logitsLast
  ) {
    MemorySegment newNPastSeg = arena.allocate(ValueLayout.JAVA_INT);
    newNPastSeg.set(ValueLayout.JAVA_INT, 0, nPast);

    int ret = LlamaRuntime.mtmd_helper_eval_chunks(
      segment,
      llamaContext.segment,
      chunks.segment(),
      nPast,
      seqId,
      nBatch,
      logitsLast,
      newNPastSeg
    );

    if (ret != 0) {
      throw new LlamaException(
        "Failed to evaluate multimodal chunks, error code: " + ret
      );
    }

    return newNPastSeg.get(ValueLayout.JAVA_INT, 0);
  }

  /**
   * Creates a bitmap from raw encoded file bytes (PNG, JPEG, WAV, etc.)
   * using the native stb_image / miniaudio decoders. This matches the exact
   * behavior of the reference llama.cpp server's {@code process_mtmd_prompt}.
   *
   * @param arena The memory arena for the buffer allocation
   * @param fileBytes The encoded file bytes (PNG, JPEG, WAV, etc.)
   * @return The native mtmd_bitmap pointer
   * @throws LlamaException if the file bytes cannot be decoded
   */
  public MemorySegment bitmapInitFromBuf(Arena arena, byte[] fileBytes) {
    MemorySegment buf = arena.allocate(fileBytes.length);
    MemorySegment.copy(
      fileBytes,
      0,
      buf,
      ValueLayout.JAVA_BYTE,
      0,
      fileBytes.length
    );
    MemorySegment bitmap = mtmd_helper_bitmap_init_from_buf(
      segment,
      buf,
      fileBytes.length
    );
    if (bitmap == null || bitmap.address() == 0) {
      throw new LlamaException(
        "Failed to create bitmap from file bytes (unsupported format or corrupt data)"
      );
    }
    return bitmap;
  }

  public MemorySegment tokenize(
    Arena arena,
    String text,
    boolean addSpecial,
    boolean parseSpecial,
    List<MtmdMedia> media
  ) {
    MemorySegment outputChunks = mtmd_input_chunks_init();

    MemorySegment textSegment = arena.allocateFrom(text);
    MemorySegment nativeMtmdInputText = LlamaRuntime.mtmd_input_text_allocate(
      arena
    );
    LlamaRuntime.mtmd_input_text_set_text(nativeMtmdInputText, textSegment);
    LlamaRuntime.mtmd_input_text_set_add_special(
      nativeMtmdInputText,
      addSpecial
    );
    LlamaRuntime.mtmd_input_text_set_parse_special(
      nativeMtmdInputText,
      parseSpecial
    );

    MemorySegment bitmapArray = arena.allocate(
      ValueLayout.ADDRESS.byteSize() * media.size()
    );
    for (int i = 0; i < media.size(); i++) {
      bitmapArray.setAtIndex(
        ValueLayout.ADDRESS,
        i,
        media.get(i).getMemorySegment()
      );
    }

    int result = mtmd_tokenize(
      segment,
      outputChunks,
      nativeMtmdInputText,
      bitmapArray,
      media.size()
    );

    if (result != 0) {
      throw new LlamaException(
        "Failed to tokenize multimodal input, error code: " + result
      );
    }
    return outputChunks;
  }

  @Override
  public void free() {
    if (!isFree()) {
      mtmd_free(segment);
      super.markFreed();
    }
  }

  @Override
  public boolean isFree() {
    return segment == null || segment.address() == 0;
  }
}
