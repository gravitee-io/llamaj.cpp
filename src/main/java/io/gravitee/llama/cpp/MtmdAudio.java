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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * Represents audio that can be processed by the multimodal model.
 * Wraps the native {@code mtmd_bitmap} structure initialized for audio data.
 */
public class MtmdAudio implements MtmdMedia {

  private MemorySegment bitmapSegment;
  private final Arena arena;

  private MtmdAudio(Arena arena, MemorySegment bitmapSegment) {
    this.arena = arena;
    this.bitmapSegment = bitmapSegment;
  }

  /**
   * Creates an MtmdAudio from raw encoded audio bytes (WAV, etc.).
   *
   * <p>This is the primary factory method. All other factory methods delegate here.
   *
   * @param arena The memory arena for allocations
   * @param audioBytes The encoded audio data (e.g., WAV file bytes)
   * @param targetSampleRate The target sample rate (e.g., 16000 for Whisper models)
   * @return A new MtmdAudio instance
   * @throws IOException If the bytes cannot be decoded
   * @throws UnsupportedAudioFileException If the audio format is not supported
   */
  public static MtmdAudio fromBytes(
    Arena arena,
    byte[] audioBytes,
    int targetSampleRate
  ) throws IOException, UnsupportedAudioFileException {
    float[] audioSamples = AudioLoader.loadAudio(audioBytes, targetSampleRate);
    return fromSamples(arena, audioSamples);
  }

  /**
   * Creates an MtmdAudio from a file path.
   *
   * @param arena The memory arena for allocations
   * @param audioPath The path to the audio file
   * @param targetSampleRate The target sample rate (e.g., 16000 for Whisper models)
   * @return A new MtmdAudio instance
   * @throws IOException If the file cannot be read
   * @throws UnsupportedAudioFileException If the audio format is not supported
   */
  public static MtmdAudio fromFile(
    Arena arena,
    Path audioPath,
    int targetSampleRate
  ) throws IOException, UnsupportedAudioFileException {
    byte[] audioBytes = Files.readAllBytes(audioPath);
    return fromBytes(arena, audioBytes, targetSampleRate);
  }

  /**
   * Creates an MtmdAudio from pre-decoded PCM float32 samples.
   *
   * @param arena The memory arena for allocations
   * @param audioSamples PCM float32 samples in range -1.0 to 1.0, mono
   * @return A new MtmdAudio instance
   */
  public static MtmdAudio fromSamples(Arena arena, float[] audioSamples) {
    if (audioSamples.length == 0) {
      throw new LlamaException("Audio contains no samples");
    }

    long nSamples = audioSamples.length;
    MemorySegment audioData = arena.allocate(
      ValueLayout.JAVA_FLOAT.byteSize() * nSamples,
      ValueLayout.JAVA_FLOAT.byteSize()
    );

    for (int i = 0; i < audioSamples.length; i++) {
      audioData.setAtIndex(ValueLayout.JAVA_FLOAT, i, audioSamples[i]);
    }

    MemorySegment bitmapSegment = LlamaRuntime.mtmd_bitmap_init_from_audio(
      nSamples,
      audioData
    );

    if (bitmapSegment.address() == 0) {
      throw new LlamaException("Failed to initialize mtmd_bitmap for audio");
    }

    // Set bitmap ID using FNV-1a hash of audio sample data for KV cache tracking.
    // This matches the reference llama.cpp server behavior (process_mtmd_prompt).
    byte[] sampleBytes = new byte[audioSamples.length * Float.BYTES];
    for (int i = 0; i < audioSamples.length; i++) {
      int bits = Float.floatToRawIntBits(audioSamples[i]);
      int offset = i * Float.BYTES;
      sampleBytes[offset] = (byte) (bits & 0xFF);
      sampleBytes[offset + 1] = (byte) ((bits >> 8) & 0xFF);
      sampleBytes[offset + 2] = (byte) ((bits >> 16) & 0xFF);
      sampleBytes[offset + 3] = (byte) ((bits >> 24) & 0xFF);
    }
    String hash = fnvHash(sampleBytes);
    MemorySegment hashSegment = arena.allocateFrom(
      hash,
      StandardCharsets.UTF_8
    );
    LlamaRuntime.mtmd_bitmap_set_id(bitmapSegment, hashSegment);

    return new MtmdAudio(arena, bitmapSegment);
  }

  /**
   * Computes an FNV-1a hash of the given byte array, matching the reference
   * llama.cpp server's bitmap ID generation for KV cache tracking.
   */
  private static String fnvHash(byte[] data) {
    long hash = 0xcbf29ce484222325L; // FNV offset basis
    for (byte b : data) {
      hash ^= (b & 0xFF);
      hash *= 0x100000001b3L; // FNV prime
    }
    return Long.toHexString(hash);
  }

  @Override
  public MemorySegment getMemorySegment() {
    return bitmapSegment;
  }

  @Override
  public void free() {
    if (bitmapSegment != null && bitmapSegment.address() != 0) {
      LlamaRuntime.mtmd_bitmap_free(bitmapSegment);
      bitmapSegment = null;
    }
  }

  @Override
  public boolean isFree() {
    return bitmapSegment == null || bitmapSegment.address() == 0;
  }
}
