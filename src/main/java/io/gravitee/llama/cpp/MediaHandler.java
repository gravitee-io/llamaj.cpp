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
import java.util.List;

/**
 * Interface for loading media into native mtmd_bitmap representations.
 *
 * <p>The type parameter {@code T} represents the input source type:
 * <ul>
 *   <li>{@code Path} -- load from a file</li>
 *   <li>{@code byte[]} -- load from raw encoded bytes (e.g., from base64, HTTP response, database)</li>
 *   <li>Any custom type the community needs</li>
 * </ul>
 *
 * <p>Built-in implementations:
 * <ul>
 *   <li>{@link ImageHandler} -- images (jpg, png, gif, bmp, webp, tiff) from {@code Path} or {@code byte[]}</li>
 *   <li>{@link AudioHandler} -- audio (wav) from {@code Path} or {@code byte[]}</li>
 * </ul>
 *
 * <p>A handler returns a list to allow a single input to produce multiple media items.
 * For images and audio, the list will typically contain a single element.
 *
 * @param <T> The input source type
 */
public interface MediaHandler<T> {
  /**
   * Checks whether this handler can process the given input.
   *
   * @param input The media source (e.g., a Path, byte[], URL)
   * @return true if this handler supports the input
   */
  boolean supports(T input);

  /**
   * Loads media from the given input and returns one or more {@link MtmdMedia} items.
   *
   * <p>Each returned media item corresponds to one media marker in the prompt.
   * The caller is responsible for freeing the returned media items.
   *
   * @param arena The memory arena for native allocations
   * @param input The media source
   * @param mtmdContext The multimodal context (provides model capabilities and audio bitrate)
   * @return A list of media items ready for multimodal processing
   * @throws Exception If the input cannot be loaded or processed
   */
  List<MtmdMedia> load(Arena arena, T input, MtmdContext mtmdContext)
    throws Exception;
}
