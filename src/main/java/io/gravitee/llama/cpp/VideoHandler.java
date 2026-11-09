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
import java.util.Locale;
import java.util.Set;

/**
 * Default {@link MediaHandler} for video files, by extension: loads them as an {@link MtmdVideo}
 * with {@link MtmdVideo.Options#defaults()}.
 */
public class VideoHandler implements MediaHandler<Object> {

  private static final Set<String> EXTENSIONS = Set.of(
    "mp4",
    "m4v",
    "mov",
    "mkv",
    "webm",
    "avi"
  );

  @Override
  public boolean supports(Object input) {
    if (input instanceof Path path) {
      String name = path.getFileName().toString();
      int dot = name.lastIndexOf('.');
      return (
        dot >= 0 &&
        EXTENSIONS.contains(name.substring(dot + 1).toLowerCase(Locale.ROOT))
      );
    }
    return false;
  }

  @Override
  public List<MtmdMedia> load(
    Arena arena,
    Object input,
    MtmdContext mtmdContext
  ) throws IOException {
    if (!mtmdContext.capabilities().video()) {
      throw new LlamaException(
        "Video input detected but the model or the llama.cpp build does not support video"
      );
    }
    if (input instanceof Path path) {
      return List.of(
        MtmdVideo.fromFile(
          arena,
          mtmdContext,
          path,
          MtmdVideo.Options.defaults()
        )
      );
    }
    throw new LlamaException(
      "VideoHandler does not support input type: " + input.getClass().getName()
    );
  }
}
