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

import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;

/**
 * Factory for creating {@link MtmdMedia} instances from various sources.
 *
 * <p>Delegates to {@link MediaHandler} implementations to detect and load media.
 * The built-in handlers are tried in order: {@link AudioHandler},
 * {@link ImageHandler} (image is the fallback).
 *
 * <p>Supports any input type that a handler can process. For example:
 * <pre>{@code
 * // From a file path (uses built-in handlers)
 * List<MtmdMedia> media = MtmdMediaFactory.fromFile(arena, path, ctx);
 *
 * // From raw bytes (e.g., base64-decoded, HTTP response)
 * byte[] imageBytes = Base64.getDecoder().decode(base64String);
 * List<MtmdMedia> media = MtmdMediaFactory.from(arena, imageBytes, ctx);
 *
 * // With a custom handler
 * MediaHandler<URL> urlHandler = new MyUrlHandler();
 * List<MtmdMedia> media = urlHandler.load(arena, new URL("https://..."), ctx);
 * }</pre>
 */
public class MtmdMediaFactory {

  @SuppressWarnings("rawtypes")
  private static final List<MediaHandler> DEFAULT_HANDLERS = List.of(
    new AudioHandler(),
    new ImageHandler()
  );

  private MtmdMediaFactory() {}

  /**
   * Loads media from a file path using the built-in handlers.
   *
   * @param arena The memory arena for allocations
   * @param mediaPath The path to the media file
   * @param mtmdContext The multimodal context
   * @return A list of media items
   * @throws Exception If the file cannot be loaded
   */
  public static List<MtmdMedia> fromFile(
    Arena arena,
    Path mediaPath,
    MtmdContext mtmdContext
  ) throws Exception {
    return from(arena, mediaPath, mtmdContext, DEFAULT_HANDLERS);
  }

  /**
   * Loads media from any source using the built-in handlers.
   *
   * <p>The input can be a {@link Path}, {@code byte[]}, or any type supported
   * by the built-in handlers.
   *
   * @param arena The memory arena for allocations
   * @param input The media source (Path, byte[], etc.)
   * @param mtmdContext The multimodal context
   * @return A list of media items
   * @throws Exception If the input cannot be loaded
   */
  public static List<MtmdMedia> from(
    Arena arena,
    Object input,
    MtmdContext mtmdContext
  ) throws Exception {
    return from(arena, input, mtmdContext, DEFAULT_HANDLERS);
  }

  /**
   * Loads media from any source using the provided handlers.
   *
   * <p>Handlers are tried in order; the first one whose {@code supports()} returns
   * true is used. If no handler matches and the input is a {@link Path}, falls back
   * to loading as an image. Otherwise throws.
   *
   * @param arena The memory arena for allocations
   * @param input The media source
   * @param mtmdContext The multimodal context
   * @param handlers The handlers to try, in order
   * @return A list of media items
   * @throws Exception If the input cannot be loaded
   */
  @SuppressWarnings({ "rawtypes", "unchecked" })
  public static List<MtmdMedia> from(
    Arena arena,
    Object input,
    MtmdContext mtmdContext,
    List<MediaHandler> handlers
  ) throws Exception {
    for (MediaHandler handler : handlers) {
      if (handler.supports(input)) {
        return handler.load(arena, input, mtmdContext);
      }
    }
    // Fallback for Path: treat as image
    if (input instanceof Path path) {
      return List.of(MtmdImage.fromFile(arena, path));
    }
    // Fallback for byte[]: treat as image
    if (input instanceof byte[] bytes) {
      return List.of(MtmdImage.fromBytes(arena, bytes));
    }
    throw new LlamaException(
      "No handler supports input of type: " + input.getClass().getName()
    );
  }
}
