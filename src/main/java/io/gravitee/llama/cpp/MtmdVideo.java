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
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A video that a vision model reads as frames, with timestamp text between them. It stands for one
 * media marker of a prompt, like an {@link MtmdImage}; the frames are decoded lazily while the
 * prompt is tokenized.
 * <p>
 * Decoding needs {@code ffmpeg} and {@code ffprobe} (on the {@code PATH}, or in
 * {@link Options#ffmpegBinDir()}) and a llama.cpp build with video support
 * ({@link MtmdCapabilities#video()}). A video can be tokenized once; load it again for another
 * prompt.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class MtmdVideo implements MtmdMedia {

  /**
   * How a video is sampled. Any {@code null} field takes llama.cpp's default.
   *
   * @param fpsTarget           Frames read per second; {@code <= 0} reads the video's own rate;
   *                            {@code null} -> 4
   * @param ffmpegBinDir        Directory holding {@code ffmpeg} and {@code ffprobe};
   *                            {@code null} -> the {@code PATH}
   * @param timestampIntervalMs Interval between timestamp texts; {@code <= 0} writes none;
   *                            {@code null} -> 5000
   * @param noSoundtrack        Frames only, for models that also hear the soundtrack;
   *                            {@code null} -> false
   */
  public record Options(
    Float fpsTarget,
    Path ffmpegBinDir,
    Long timestampIntervalMs,
    Boolean noSoundtrack
  ) {
    public static Options defaults() {
      return new Options(null, null, null, null);
    }

    public Options withFpsTarget(float fpsTarget) {
      return new Options(
        fpsTarget,
        ffmpegBinDir,
        timestampIntervalMs,
        noSoundtrack
      );
    }

    public Options withFfmpegBinDir(Path ffmpegBinDir) {
      return new Options(
        fpsTarget,
        ffmpegBinDir,
        timestampIntervalMs,
        noSoundtrack
      );
    }

    public Options withTimestampIntervalMs(long timestampIntervalMs) {
      return new Options(
        fpsTarget,
        ffmpegBinDir,
        timestampIntervalMs,
        noSoundtrack
      );
    }

    public Options withNoSoundtrack(boolean noSoundtrack) {
      return new Options(
        fpsTarget,
        ffmpegBinDir,
        timestampIntervalMs,
        noSoundtrack
      );
    }
  }

  /**
   * The size and sampling of a video.
   *
   * @param width   Frame width in pixels
   * @param height  Frame height in pixels
   * @param fps     Frames read per second
   * @param nFrames Estimated frames read, {@code -1} if unknown
   */
  public record Info(int width, int height, float fps, int nFrames) {}

  private MemorySegment bitmap;
  private MemorySegment video;
  private final Info info;

  private MtmdVideo(MemorySegment bitmap, MemorySegment video, Info info) {
    this.bitmap = bitmap;
    this.video = video;
    this.info = info;
  }

  /**
   * Loads a video file.
   *
   * @param arena   Arena for the allocations that live as long as the video
   * @param mtmd    The multimodal context of the vision model
   * @param path    The video file
   * @param options Sampling options; pass {@link Options#defaults()} for llama.cpp's defaults
   * @throws IOException    if the file cannot be read
   * @throws LlamaException if the bytes are not a video, or ffmpeg cannot decode them
   */
  public static MtmdVideo fromFile(
    Arena arena,
    MtmdContext mtmd,
    Path path,
    Options options
  ) throws IOException {
    return fromBytes(arena, mtmd, Files.readAllBytes(path), options);
  }

  /**
   * Loads a video from its encoded bytes (mp4, webm, mkv…).
   *
   * @param arena   Arena for the allocations that live as long as the video
   * @param mtmd    The multimodal context of the vision model
   * @param bytes   The encoded video
   * @param options Sampling options; pass {@link Options#defaults()} for llama.cpp's defaults
   * @throws LlamaException if the bytes are not a video, or ffmpeg cannot decode them
   */
  public static MtmdVideo fromBytes(
    Arena arena,
    MtmdContext mtmd,
    byte[] bytes,
    Options options
  ) {
    MemorySegment opt = LlamaRuntime.mtmd_helper_init_opt_default(arena);
    MemorySegment params = LlamaRuntime.mtmd_helper_init_opt_video_params(opt);
    if (options.fpsTarget() != null) {
      LlamaRuntime.mtmd_helper_video_init_params_set_fps_target(
        params,
        options.fpsTarget()
      );
    }
    if (options.ffmpegBinDir() != null) {
      LlamaRuntime.mtmd_helper_video_init_params_set_ffmpeg_bin_dir(
        params,
        arena.allocateFrom(options.ffmpegBinDir().toString())
      );
    }
    if (options.timestampIntervalMs() != null) {
      LlamaRuntime.mtmd_helper_video_init_params_set_timestamp_interval_ms(
        params,
        options.timestampIntervalMs()
      );
    }
    if (options.noSoundtrack() != null) {
      LlamaRuntime.mtmd_helper_video_init_params_set_no_soundtrack(
        params,
        options.noSoundtrack()
      );
    }

    MemorySegment wrapper;
    try (var local = Arena.ofConfined()) {
      MemorySegment buf = local.allocate(bytes.length);
      MemorySegment.copy(bytes, 0, buf, ValueLayout.JAVA_BYTE, 0, bytes.length);
      wrapper = LlamaRuntime.mtmd_helper_bitmap_init_from_buf(
        arena,
        mtmd.segment,
        buf,
        bytes.length,
        false,
        opt
      );
    }
    MemorySegment bitmap = wrapper.get(ValueLayout.ADDRESS, 0);
    MemorySegment video = wrapper.get(
      ValueLayout.ADDRESS,
      ValueLayout.ADDRESS.byteSize()
    );
    if (bitmap == null || bitmap.address() == 0) {
      throw new LlamaException(
        "failed to load the video: unsupported data, or ffmpeg/ffprobe not found"
      );
    }
    if (video == null || video.address() == 0) {
      LlamaRuntime.mtmd_bitmap_free(bitmap);
      throw new LlamaException(
        "the bytes are not a video, load them as an image or audio"
      );
    }
    return new MtmdVideo(bitmap, video, readInfo(video));
  }

  private static Info readInfo(MemorySegment video) {
    try (var local = Arena.ofConfined()) {
      MemorySegment info = LlamaRuntime.mtmd_helper_video_get_info(
        local,
        video
      );
      return new Info(
        LlamaRuntime.<Integer>mtmd_helper_video_info_field(info, "width"),
        LlamaRuntime.<Integer>mtmd_helper_video_info_field(info, "height"),
        LlamaRuntime.<Float>mtmd_helper_video_info_field(info, "fps"),
        LlamaRuntime.<Integer>mtmd_helper_video_info_field(info, "n_frames")
      );
    }
  }

  /** The size and sampling of the video. */
  public Info info() {
    return info;
  }

  @Override
  public MemorySegment getMemorySegment() {
    return bitmap;
  }

  @Override
  public void free() {
    if (bitmap != null) {
      LlamaRuntime.mtmd_bitmap_free(bitmap);
      bitmap = null;
    }
    if (video != null) {
      LlamaRuntime.mtmd_helper_video_free(video);
      video = null;
    }
  }

  @Override
  public boolean isFree() {
    return bitmap == null && video == null;
  }
}
