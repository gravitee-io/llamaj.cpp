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

import java.util.List;
import java.util.Objects;

/**
 * A prompt with images or audio for {@link LabelDecider} or {@link PooledDecider}: rendered text
 * holding one media marker per item of {@code media} (the marker of the decider's
 * {@link MtmdContext}, {@link MtmdContext#marker()}), tokenized with the media by the decider.
 * <p>
 * The decider must have been created with an {@link MtmdContext}.
 *
 * @param text     The rendered prompt text, with the media markers
 * @param media    The images or audio, in marker order
 * @param nOptions The number of scores to return
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record MediaPrompt(
  String text,
  List<MtmdMedia> media,
  int nOptions
) implements DecisionPrompt {
  public MediaPrompt {
    Objects.requireNonNull(text, "text");
    media = List.copyOf(Objects.requireNonNull(media, "media"));
    if (nOptions < 1) {
      throw new IllegalArgumentException("nOptions must be at least 1");
    }
  }

  /** A prompt with media. */
  public static MediaPrompt of(
    String text,
    List<MtmdMedia> media,
    int nOptions
  ) {
    return new MediaPrompt(text, media, nOptions);
  }
}
