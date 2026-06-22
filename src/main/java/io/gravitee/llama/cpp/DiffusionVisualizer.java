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

import java.nio.charset.StandardCharsets;

/**
 * A {@link DiffusionGenerator.StepCallback} that renders the canvas in place on each
 * denoising step, so the diffusion process is visible in the terminal — the Java analogue
 * of llama.cpp's {@code --diffusion-visual} mode.
 *
 * <p>Each step it redraws the whole canvas: still-masked positions show as a placeholder
 * glyph, unmasked positions are detokenized to text, and positions that were just filled
 * on this step are highlighted. The screen is cleared and the cursor homed between frames
 * (ANSI), so successive frames overwrite one another into an animation.
 *
 * <pre>{@code
 * var viz = new DiffusionVisualizer(vocab, maskToken);
 * new DiffusionGenerator(context).generate(prompt, params, viz);
 * }</pre>
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class DiffusionVisualizer
  implements DiffusionGenerator.StepCallback {

  private static final String CLEAR_HOME = "\033[2J\033[H";
  private static final String DIM = "\033[2m";
  private static final String GREEN = "\033[32m";
  private static final String BOLD = "\033[1m";
  private static final String RESET = "\033[0m";

  private final LlamaVocab vocab;
  private final int maskToken;
  private final int promptLength;
  private final String maskGlyph;
  private final boolean color;

  private int[] previous;

  public DiffusionVisualizer(LlamaVocab vocab, int maskToken) {
    this(vocab, maskToken, 0, "·", true);
  }

  /**
   * @param vocab        Vocab used to detokenize filled positions
   * @param maskToken    The model's mask token id
   * @param promptLength Number of leading positions that are the (fixed) prompt; rendered
   *                     dim so it's clear the prompt is context, not generation
   * @param maskGlyph    Placeholder rendered for still-masked positions (e.g. {@code "·"})
   * @param color        Whether to emit ANSI colour / clear-screen codes
   */
  public DiffusionVisualizer(
    LlamaVocab vocab,
    int maskToken,
    int promptLength,
    String maskGlyph,
    boolean color
  ) {
    this.vocab = vocab;
    this.maskToken = maskToken;
    this.promptLength = promptLength;
    this.maskGlyph = maskGlyph;
    this.color = color;
  }

  @Override
  public boolean onStep(int step, int totalSteps, int[] tokens) {
    var sb = new StringBuilder();
    if (color) {
      sb.append(CLEAR_HOME);
    }
    // Only the generated region is rendered — the prompt is fixed context, not output.
    int genLength = tokens.length - promptLength;
    int filled = 0;
    for (int i = promptLength; i < tokens.length; i++) {
      if (tokens[i] != maskToken) {
        filled++;
      }
    }
    sb
      .append(wrap(BOLD, "diffusion"))
      .append("  step ")
      .append(step + 1)
      .append('/')
      .append(totalSteps)
      .append("   filled ")
      .append(filled)
      .append('/')
      .append(genLength)
      .append("\n\n");

    for (int i = promptLength; i < tokens.length; i++) {
      int t = tokens[i];
      if (t == maskToken) {
        sb.append(wrap(DIM, maskGlyph));
        continue;
      }
      String piece = new String(vocab.tokenToPiece(t), StandardCharsets.UTF_8);
      boolean justFilled = previous != null && previous[i] == maskToken;
      sb.append(justFilled ? wrap(GREEN, piece) : piece);
    }
    sb.append('\n');
    System.out.print(sb);
    System.out.flush();

    previous = tokens.clone();
    return true; // never abort
  }

  private String wrap(String code, String text) {
    return color ? code + text + RESET : text;
  }
}
