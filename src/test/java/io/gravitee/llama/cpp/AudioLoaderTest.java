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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Path;
import javax.sound.sampled.UnsupportedAudioFileException;
import org.junit.jupiter.api.Test;

/**
 * Tests for AudioLoader utility class.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class AudioLoaderTest {

  @Test
  void should_detect_wav_format() {
    Path audioFile = Path.of("src/test/resources/test-2.wav");
    AudioFormat format = AudioLoader.detectFormat(audioFile);
    assertThat(format).isEqualTo(AudioFormat.WAV);
  }

  @Test
  void should_load_wav_audio_file()
    throws IOException, UnsupportedAudioFileException {
    Path audioFile = Path.of("src/test/resources/test-2.wav");

    // Load with standard audio sample rate
    float[] samples = AudioLoader.loadAudio(audioFile, 16000);

    assertThat(samples).isNotNull();
    assertThat(samples.length).isGreaterThan(0);

    // Verify samples are in valid range (-1.0 to 1.0)
    for (float sample : samples) {
      assertThat(sample).isBetween(-1.0f, 1.0f);
    }
  }

  @Test
  void should_load_audio_with_different_sample_rates()
    throws IOException, UnsupportedAudioFileException {
    Path audioFile = Path.of("src/test/resources/test-2.wav");

    // Test multiple sample rates
    for (int sampleRate : new int[] { 8000, 16000, 22050, 44100 }) {
      float[] samples = AudioLoader.loadAudio(audioFile, sampleRate);
      assertThat(samples).isNotNull();
      assertThat(samples.length).isGreaterThan(0);
    }
  }

  @Test
  void should_throw_on_nonexistent_file() {
    Path audioFile = Path.of("nonexistent-audio-file.mp3");

    assertThatThrownBy(() ->
      AudioLoader.loadAudio(audioFile, 16000)
    ).isInstanceOf(IOException.class);
  }

  @Test
  void should_detect_format_by_extension() {
    Path wavFile = Path.of("audio.wav");
    AudioFormat format = AudioLoader.detectFormat(wavFile);
    assertThat(format).isEqualTo(AudioFormat.WAV);
  }

  @Test
  void should_return_null_for_unknown_extension() {
    Path unknownFile = Path.of("audio.xyz");
    AudioFormat format = AudioLoader.detectFormat(unknownFile);
    assertThat(format).isNull();
  }

  @Test
  void should_handle_case_insensitive_extension() {
    Path wavFile = Path.of("audio.WAV");
    AudioFormat format = AudioLoader.detectFormat(wavFile);
    assertThat(format).isEqualTo(AudioFormat.WAV);
  }
}
