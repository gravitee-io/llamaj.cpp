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

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/**
 * Represents an image that can be processed by the multimodal model.
 * Wraps the native {@code mtmd_bitmap} structure.
 */
public class MtmdImage implements MtmdMedia {

  private MemorySegment bitmapSegment;
  private final Arena arena;

  private MtmdImage(Arena arena, MemorySegment bitmapSegment) {
    this.arena = arena;
    this.bitmapSegment = bitmapSegment;
  }

  /**
   * Creates an MtmdImage from raw encoded image bytes (JPEG, PNG, etc.)
   * using the native stb_image decoder. This produces byte-for-byte identical
   * results to the reference llama.cpp server.
   *
   * <p>This is the preferred factory method for production use. It delegates
   * image decoding to the native {@code mtmd_helper_bitmap_init_from_buf}
   * which uses stb_image internally, guaranteeing identical pixel values
   * to the reference implementation.
   *
   * @param arena The memory arena for allocations
   * @param mtmdContext The multimodal context (needed by the native helper)
   * @param imageBytes The encoded image data (e.g., JPEG, PNG bytes)
   * @return A new MtmdImage instance
   */
  public static MtmdImage fromBytesNative(
    Arena arena,
    MtmdContext mtmdContext,
    byte[] imageBytes
  ) {
    MemorySegment bitmapSegment = mtmdContext.bitmapInitFromBuf(
      arena,
      imageBytes
    );

    // Set bitmap ID using FNV-1a hash of the raw file bytes for KV cache tracking.
    String hash = fnvHash(imageBytes);
    MemorySegment hashSegment = arena.allocateFrom(
      hash,
      StandardCharsets.UTF_8
    );
    LlamaRuntime.mtmd_bitmap_set_id(bitmapSegment, hashSegment);

    return new MtmdImage(arena, bitmapSegment);
  }

  /**
   * Creates an MtmdImage from raw encoded image bytes (JPEG, PNG, etc.).
   *
   * <p>This is the primary factory method. All other factory methods delegate here.
   *
   * @param arena The memory arena for allocations
   * @param imageBytes The encoded image data (e.g., JPEG, PNG bytes)
   * @return A new MtmdImage instance
   * @throws IOException If the bytes cannot be decoded as an image
   */
  public static MtmdImage fromBytes(Arena arena, byte[] imageBytes)
    throws IOException {
    BufferedImage image = ImageIO.read(new ByteArrayInputStream(imageBytes));
    if (image == null) {
      throw new IOException("Could not decode image from provided bytes");
    }
    return fromBufferedImage(arena, image);
  }

  /**
   * Creates an MtmdImage from a file path.
   *
   * @param arena The memory arena for allocations
   * @param imagePath The path to the image file
   * @return A new MtmdImage instance
   * @throws IOException If the file cannot be read or decoded
   */
  public static MtmdImage fromFile(Arena arena, Path imagePath)
    throws IOException {
    byte[] imageBytes = Files.readAllBytes(imagePath);
    return fromBytes(arena, imageBytes);
  }

  /**
   * Creates an MtmdImage from a {@link BufferedImage}.
   *
   * @param arena The memory arena for allocations
   * @param image The decoded image
   * @return A new MtmdImage instance
   */
  public static MtmdImage fromBufferedImage(Arena arena, BufferedImage image) {
    int width = image.getWidth();
    int height = image.getHeight();

    // Force conversion to TYPE_INT_RGB to ensure no alpha pre-multiplication issues.
    // Java's BufferedImage may use pre-multiplied alpha (TYPE_INT_ARGB_PRE, TYPE_4BYTE_ABGR_PRE)
    // which corrupts RGB values for semi-transparent pixels. Drawing onto a TYPE_INT_RGB
    // target composites alpha properly and produces clean RGB values, matching stb_image behavior.
    if (image.getType() != BufferedImage.TYPE_INT_RGB) {
      BufferedImage rgbImage = new BufferedImage(
        width,
        height,
        BufferedImage.TYPE_INT_RGB
      );
      Graphics2D g = rgbImage.createGraphics();
      g.drawImage(image, 0, 0, null);
      g.dispose();
      image = rgbImage;
    }

    // Convert image to RGB (3 bytes per pixel)
    byte[] rgbData = new byte[width * height * 3];
    int pixelIndex = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int pixel = image.getRGB(x, y);
        rgbData[pixelIndex++] = (byte) ((pixel >> 16) & 0xFF); // Red
        rgbData[pixelIndex++] = (byte) ((pixel >> 8) & 0xFF); // Green
        rgbData[pixelIndex++] = (byte) (pixel & 0xFF); // Blue
      }
    }

    MemorySegment imageData = arena.allocate(rgbData.length, 1);
    MemorySegment.copy(
      rgbData,
      0,
      imageData,
      ValueLayout.JAVA_BYTE,
      0,
      rgbData.length
    );

    MemorySegment bitmapSegment = LlamaRuntime.mtmd_bitmap_init(
      width,
      height,
      imageData
    );

    if (bitmapSegment.address() == 0) {
      throw new LlamaException("Failed to initialize mtmd_bitmap for image.");
    }

    // Set bitmap ID using FNV-1a hash of pixel data for KV cache tracking.
    // This matches the reference llama.cpp server behavior (process_mtmd_prompt).
    String hash = fnvHash(rgbData);
    MemorySegment hashSegment = arena.allocateFrom(
      hash,
      StandardCharsets.UTF_8
    );
    LlamaRuntime.mtmd_bitmap_set_id(bitmapSegment, hashSegment);

    return new MtmdImage(arena, bitmapSegment);
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
