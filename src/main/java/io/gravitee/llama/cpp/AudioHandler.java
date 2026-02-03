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
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * Default {@link MediaHandler} for audio files and audio bytes.
 *
 * <p>Accepts:
 * <ul>
 *   <li>{@link Path} -- loads from a file</li>
 *   <li>{@code byte[]} -- decodes from raw audio bytes (WAV)</li>
 * </ul>
 *
 * <p>Supports WAV audio natively via the Java Sound API.
 *
 * <p>To add support for additional audio formats (MP3, FLAC, etc.), either:
 * <ul>
 *   <li>Convert to WAV beforehand (e.g., using ffmpeg or afconvert)</li>
 *   <li>Implement a custom {@link MediaHandler} with a third-party decoder</li>
 * </ul>
 */
public class AudioHandler implements MediaHandler<Object> {

  private static final Set<String> EXTENSIONS = Set.of("wav", "wave");

  @Override
  public boolean supports(Object input) {
    if (input instanceof Path path) {
      String ext = getExtension(path);
      if (ext != null && EXTENSIONS.contains(ext)) {
        return true;
      }
      AudioFormat format = AudioLoader.detectFormat(path);
      return format == AudioFormat.WAV;
    }
    // byte[] cannot be reliably detected as audio vs image without magic bytes,
    // so we don't claim support for raw bytes by default.
    // Use this handler explicitly for known audio bytes.
    return false;
  }

  @Override
  public List<MtmdMedia> load(
    Arena arena,
    Object input,
    MtmdContext mtmdContext
  ) throws IOException, UnsupportedAudioFileException {
    if (!mtmdContext.supportsAudio()) {
      throw new LlamaException(
        "Audio input detected but model does not support audio"
      );
    }
    int sampleRate = mtmdContext.getAudioBitrate();
    if (sampleRate <= 0) {
      throw new LlamaException("Model does not provide a valid audio bitrate");
    }

    if (input instanceof Path path) {
      return List.of(MtmdAudio.fromFile(arena, path, sampleRate));
    }
    if (input instanceof byte[] bytes) {
      return List.of(MtmdAudio.fromBytes(arena, bytes, sampleRate));
    }
    throw new LlamaException(
      "AudioHandler does not support input type: " + input.getClass().getName()
    );
  }

  private static String getExtension(Path path) {
    String name = path.getFileName().toString();
    int dot = name.lastIndexOf('.');
    if (dot < 0 || dot == name.length() - 1) {
      return null;
    }
    return name.substring(dot + 1).toLowerCase();
  }
}
