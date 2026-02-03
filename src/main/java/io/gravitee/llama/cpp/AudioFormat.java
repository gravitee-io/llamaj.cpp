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
 * Supported audio file formats for multimodal audio processing.
 */
public enum AudioFormat {
  WAV("wav", new byte[] { 0x52, 0x49, 0x46, 0x46 }), // "RIFF"
  MP3("mp3", new byte[] { (byte) 0xFF, (byte) 0xFB }), // MP3 frame sync
  FLAC("flac", new byte[] { 0x66, 0x4C, 0x61, 0x43 }); // "fLaC"

  private final String extension;
  private final byte[] magicBytes;

  AudioFormat(String extension, byte[] magicBytes) {
    this.extension = extension;
    this.magicBytes = magicBytes;
  }

  /**
   * Gets the file extension for this format (without dot).
   *
   * @return The file extension
   */
  public String getExtension() {
    return extension;
  }

  /**
   * Gets the magic bytes that identify this format.
   *
   * @return The magic bytes array
   */
  public byte[] getMagicBytes() {
    return magicBytes;
  }

  /**
   * Detects audio format from file extension.
   *
   * @param filename The filename or path
   * @return The detected AudioFormat, or null if format is unknown
   */
  public static AudioFormat fromFileName(String filename) {
    if (filename == null || filename.isEmpty()) {
      return null;
    }
    String lowerName = filename.toLowerCase();
    for (AudioFormat format : values()) {
      if (lowerName.endsWith("." + format.extension)) {
        return format;
      }
    }
    return null;
  }

  /**
   * Detects audio format from file magic bytes.
   *
   * @param data The file data (at least first 4 bytes required)
   * @return The detected AudioFormat, or null if format is unknown
   */
  public static AudioFormat fromMagicBytes(byte[] data) {
    if (data == null || data.length < 2) {
      return null;
    }

    // Check WAV (RIFF....WAVE pattern)
    if (
      data.length >= 12 &&
      matches(data, 0, WAV.magicBytes) &&
      data.length >= 12 &&
      data[8] == 0x57 &&
      data[9] == 0x41 &&
      data[10] == 0x56 &&
      data[11] == 0x45
    ) {
      return WAV;
    }

    // Check MP3 (FF FB or FF FA)
    if (data.length >= 2) {
      int first = data[0] & 0xFF;
      int second = data[1] & 0xFF;
      // MP3 sync: 0xFF followed by 0xFA or 0xFB
      if (first == 0xFF && (second == 0xFA || second == 0xFB)) {
        return MP3;
      }
    }

    // Check FLAC
    if (data.length >= 4 && matches(data, 0, FLAC.magicBytes)) {
      return FLAC;
    }

    return null;
  }

  /**
   * Checks if data at offset matches the expected bytes pattern.
   *
   * @param data The data to check
   * @param offset The offset in data
   * @param pattern The pattern to match
   * @return true if pattern matches at offset
   */
  private static boolean matches(byte[] data, int offset, byte[] pattern) {
    if (offset + pattern.length > data.length) {
      return false;
    }
    for (int i = 0; i < pattern.length; i++) {
      if (data[offset + i] != pattern[i]) {
        return false;
      }
    }
    return true;
  }
}
