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

/**
 * Default {@link MediaHandler} for image files and image bytes.
 *
 * <p>Accepts:
 * <ul>
 *   <li>{@link Path} -- loads from a file</li>
 *   <li>{@code byte[]} -- decodes from raw image bytes (JPEG, PNG, etc.)</li>
 * </ul>
 *
 * <p>Supports common image formats: JPEG, PNG, GIF, BMP, WebP, and TIFF.
 * Uses Java's {@link javax.imageio.ImageIO} to decode images into RGB pixel data.
 *
 * <p>To support additional image formats or use a different decoding library,
 * implement {@link MediaHandler} directly.
 */
public class ImageHandler implements MediaHandler<Object> {

  private static final Set<String> EXTENSIONS = Set.of(
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "webp",
    "tiff",
    "tif"
  );

  @Override
  public boolean supports(Object input) {
    if (input instanceof Path path) {
      String ext = getExtension(path);
      return ext != null && EXTENSIONS.contains(ext);
    }
    if (input instanceof byte[]) {
      return true; // byte[] always accepted; ImageIO.read will fail if not an image
    }
    return false;
  }

  @Override
  public List<MtmdMedia> load(
    Arena arena,
    Object input,
    MtmdContext mtmdContext
  ) throws IOException {
    if (input instanceof Path path) {
      return List.of(MtmdImage.fromFile(arena, path));
    }
    if (input instanceof byte[] bytes) {
      return List.of(MtmdImage.fromBytes(arena, bytes));
    }
    throw new LlamaException(
      "ImageHandler does not support input type: " + input.getClass().getName()
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
