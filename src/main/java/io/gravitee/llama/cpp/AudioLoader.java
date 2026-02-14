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

import static javax.sound.sampled.AudioFormat.Encoding.PCM_SIGNED;

import java.io.*;
import java.nio.file.Path;
import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * Utility class for loading and processing audio files.
 * Handles format detection, decoding, and resampling to target bitrate.
 */
public class AudioLoader {

  private AudioLoader() {
    // Utility class
  }

  /**
   * Loads audio from a file and converts it to PCM float32 format.
   * Automatically resamples to the target sample rate.
   *
   * @param audioPath The path to the audio file
   * @param targetSampleRate The target sample rate in Hz (e.g., 16000 for Whisper)
   * @return An array of float samples in PCM format (range -1.0 to 1.0)
   * @throws IOException If the file cannot be read or is not a valid audio format
   * @throws UnsupportedAudioFileException If the audio format is not supported
   */
  public static float[] loadAudio(Path audioPath, int targetSampleRate)
    throws IOException, UnsupportedAudioFileException {
    File audioFile = audioPath.toFile();
    try (
      AudioInputStream audioStream = AudioSystem.getAudioInputStream(audioFile)
    ) {
      return decodeAndResample(audioStream, targetSampleRate);
    }
  }

  /**
   * Loads audio from raw encoded bytes (e.g., WAV) and converts to PCM float32 format.
   * Automatically resamples to the target sample rate.
   *
   * @param audioBytes The encoded audio data
   * @param targetSampleRate The target sample rate in Hz (e.g., 16000 for Whisper)
   * @return An array of float samples in PCM format (range -1.0 to 1.0)
   * @throws IOException If the bytes cannot be decoded
   * @throws UnsupportedAudioFileException If the audio format is not supported
   */
  public static float[] loadAudio(byte[] audioBytes, int targetSampleRate)
    throws IOException, UnsupportedAudioFileException {
    try (
      AudioInputStream audioStream = AudioSystem.getAudioInputStream(
        new ByteArrayInputStream(audioBytes)
      )
    ) {
      return decodeAndResample(audioStream, targetSampleRate);
    }
  }

  private static float[] decodeAndResample(
    AudioInputStream audioStream,
    int targetSampleRate
  ) throws IOException {
    var originalFormat = audioStream.getFormat();

    // Create target audio format: PCM, mono or stereo, signed samples, target sample rate
    var targetFormat = new javax.sound.sampled.AudioFormat(
      PCM_SIGNED,
      targetSampleRate,
      16, // bits per sample
      originalFormat.getChannels(),
      originalFormat.getChannels() * 2, // frame size
      targetSampleRate,
      originalFormat.isBigEndian()
    );

    // Get a resampled audio input stream if necessary
    AudioInputStream resampledStream = AudioSystem.getAudioInputStream(
      targetFormat,
      audioStream
    );

    return readAudioSamples(resampledStream);
  }

  /**
   * Reads audio samples from an AudioInputStream and converts to float PCM format.
   *
   * @param audioStream The audio input stream
   * @return An array of float samples
   * @throws IOException If reading fails
   */
  private static float[] readAudioSamples(AudioInputStream audioStream)
    throws IOException {
    javax.sound.sampled.AudioFormat format = audioStream.getFormat();
    int frameSize = format.getFrameSize();
    int channels = format.getChannels();
    int sampleSizeInBits = format.getSampleSizeInBits();
    boolean isBigEndian = format.isBigEndian();

    // Read all audio data into memory
    ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
    byte[] buffer = new byte[4096];
    int bytesRead;
    while ((bytesRead = audioStream.read(buffer)) != -1) {
      byteStream.write(buffer, 0, bytesRead);
    }
    byte[] audioData = byteStream.toByteArray();

    // Convert byte array to float samples
    int numSamples = audioData.length / frameSize;
    float[] samples = new float[numSamples];

    for (int i = 0; i < numSamples; i++) {
      int byteOffset = i * frameSize;
      samples[i] = bytesToFloat(
        audioData,
        byteOffset,
        sampleSizeInBits,
        isBigEndian,
        channels
      );
    }

    return samples;
  }

  /**
   * Converts bytes to a float sample, averaging channels if stereo.
   *
   * @param data The byte array containing audio data
   * @param offset The offset in the byte array
   * @param sampleSizeInBits The sample size in bits (typically 16)
   * @param isBigEndian Whether the byte order is big-endian
   * @param channels Number of audio channels
   * @return A float sample in range -1.0 to 1.0
   */
  private static float bytesToFloat(
    byte[] data,
    int offset,
    int sampleSizeInBits,
    boolean isBigEndian,
    int channels
  ) {
    float sum = 0;

    for (int ch = 0; ch < channels; ch++) {
      int bytePos = offset + ((ch * sampleSizeInBits) / 8);
      float sample;

      if (sampleSizeInBits == 16) {
        int value = toInt16(data, bytePos, isBigEndian);
        sample = value / 32768.0f;
      } else if (sampleSizeInBits == 24) {
        int value = toInt24(data, bytePos, isBigEndian);
        sample = value / 8388608.0f;
      } else if (sampleSizeInBits == 8) {
        int value = (data[bytePos] & 0xFF) - 128;
        sample = value / 128.0f;
      } else {
        // Fallback for other bit depths
        sample = 0.0f;
      }

      sum += sample;
    }

    // Average channels for stereo
    return sum / channels;
  }

  /**
   * Converts two bytes to a signed 16-bit integer.
   *
   * @param data The byte array
   * @param offset The offset in the array
   * @param isBigEndian Whether byte order is big-endian
   * @return The signed integer value
   */
  private static int toInt16(byte[] data, int offset, boolean isBigEndian) {
    int value;
    if (isBigEndian) {
      value = ((data[offset] & 0xFF) << 8) | (data[offset + 1] & 0xFF);
    } else {
      value = ((data[offset + 1] & 0xFF) << 8) | (data[offset] & 0xFF);
    }
    // Sign-extend from 16-bit to 32-bit
    return (short) value;
  }

  /**
   * Converts three bytes to a signed 24-bit integer.
   *
   * @param data The byte array
   * @param offset The offset in the array
   * @param isBigEndian Whether byte order is big-endian
   * @return The signed integer value
   */
  private static int toInt24(byte[] data, int offset, boolean isBigEndian) {
    int value;
    if (isBigEndian) {
      value =
        ((data[offset] & 0xFF) << 16) |
        ((data[offset + 1] & 0xFF) << 8) |
        (data[offset + 2] & 0xFF);
    } else {
      value =
        ((data[offset + 2] & 0xFF) << 16) |
        ((data[offset + 1] & 0xFF) << 8) |
        (data[offset] & 0xFF);
    }
    // Sign-extend from 24-bit to 32-bit
    if ((value & 0x800000) != 0) {
      value |= 0xFF000000;
    }
    return value;
  }

  /**
   * Detects the audio format of a file.
   *
   * @param audioPath The path to the audio file
   * @return The detected AudioFormat, or null if unknown
   */
  public static AudioFormat detectFormat(Path audioPath) {
    // Try file extension first
    String filename = audioPath.getFileName().toString();
    AudioFormat format = AudioFormat.fromFileName(filename);
    if (format != null) {
      return format;
    }

    // Try magic bytes as fallback
    try (InputStream fis = new FileInputStream(audioPath.toFile())) {
      byte[] magicBytes = new byte[12];
      int bytesRead = fis.read(magicBytes);
      if (bytesRead > 0) {
        byte[] truncated = new byte[bytesRead];
        System.arraycopy(magicBytes, 0, truncated, 0, bytesRead);
        return AudioFormat.fromMagicBytes(truncated);
      }
    } catch (IOException e) {
      // Fallback to extension
    }

    return null;
  }
}
