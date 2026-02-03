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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests for AudioFormat enum and format detection.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class AudioFormatTest {

  @Test
  void should_detect_wav_by_filename() {
    AudioFormat format = AudioFormat.fromFileName("audio.wav");
    assertThat(format).isEqualTo(AudioFormat.WAV);
  }

  @Test
  void should_detect_mp3_by_filename() {
    AudioFormat format = AudioFormat.fromFileName("music.mp3");
    assertThat(format).isEqualTo(AudioFormat.MP3);
  }

  @Test
  void should_detect_flac_by_filename() {
    AudioFormat format = AudioFormat.fromFileName("song.flac");
    assertThat(format).isEqualTo(AudioFormat.FLAC);
  }

  @ParameterizedTest
  @ValueSource(strings = { "file.WAV", "audio.Wav", "SAMPLE.wav" })
  void should_detect_wav_case_insensitive(String filename) {
    AudioFormat format = AudioFormat.fromFileName(filename);
    assertThat(format).isEqualTo(AudioFormat.WAV);
  }

  @Test
  void should_return_null_for_unknown_extension() {
    AudioFormat format = AudioFormat.fromFileName("file.xyz");
    assertThat(format).isNull();
  }

  @Test
  void should_return_null_for_null_filename() {
    AudioFormat format = AudioFormat.fromFileName(null);
    assertThat(format).isNull();
  }

  @Test
  void should_return_null_for_empty_filename() {
    AudioFormat format = AudioFormat.fromFileName("");
    assertThat(format).isNull();
  }

  @Test
  void should_detect_wav_by_magic_bytes() {
    // RIFF magic bytes
    byte[] wavMagic = new byte[] {
      (byte) 0x52,
      (byte) 0x49,
      (byte) 0x46,
      (byte) 0x46,
      (byte) 0x00,
      (byte) 0x00,
      (byte) 0x00,
      (byte) 0x00,
      (byte) 0x57,
      (byte) 0x41,
      (byte) 0x56,
      (byte) 0x45,
    };
    AudioFormat format = AudioFormat.fromMagicBytes(wavMagic);
    assertThat(format).isEqualTo(AudioFormat.WAV);
  }

  @Test
  void should_detect_flac_by_magic_bytes() {
    // FLAC magic bytes
    byte[] flacMagic = new byte[] {
      (byte) 0x66,
      (byte) 0x4C,
      (byte) 0x61,
      (byte) 0x43,
    };
    AudioFormat format = AudioFormat.fromMagicBytes(flacMagic);
    assertThat(format).isEqualTo(AudioFormat.FLAC);
  }

  @Test
  void should_return_null_for_unknown_magic_bytes() {
    byte[] unknownMagic = new byte[] { (byte) 0xAA, (byte) 0xBB };
    AudioFormat format = AudioFormat.fromMagicBytes(unknownMagic);
    assertThat(format).isNull();
  }

  @Test
  void should_return_null_for_insufficient_magic_bytes() {
    byte[] tooShort = new byte[] { (byte) 0xFF };
    AudioFormat format = AudioFormat.fromMagicBytes(tooShort);
    assertThat(format).isNull();
  }

  @Test
  void should_return_null_for_null_magic_bytes() {
    AudioFormat format = AudioFormat.fromMagicBytes(null);
    assertThat(format).isNull();
  }

  @Test
  void should_have_correct_extension_for_wav() {
    assertThat(AudioFormat.WAV.getExtension()).isEqualTo("wav");
  }

  @Test
  void should_have_correct_extension_for_mp3() {
    assertThat(AudioFormat.MP3.getExtension()).isEqualTo("mp3");
  }

  @Test
  void should_have_correct_extension_for_flac() {
    assertThat(AudioFormat.FLAC.getExtension()).isEqualTo("flac");
  }
}
