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
import java.util.Arrays;
import java.util.Optional;

/**
 * The family of a native decision model: a classifier that scores the options of a rendered prompt
 * ({@link DecisionPrompt}) in one forward pass, without generating tokens.
 * <p>
 * A GGUF declares it with the {@code <arch>.decision.type} metadata key; a model without that key
 * is an ordinary model. Each family reads its scores differently (mirrors llama.cpp's
 * {@code common_decision_type}):
 * <ul>
 *   <li>{@link #OPENJEV} — logits of one label token per option, read at the last prompt token</li>
 *   <li>{@link #LEV} — same as openjev, noul is read from a rating scale</li>
 *   <li>{@link #KEV} — dot product of the hidden states of the last token and of one end token per option</li>
 *   <li>{@link #NIMBLE} — same as openjev, the prompt lists all the questions of the request</li>
 *   <li>{@link #LAYA} — score of one marker token per option, read from the embeddings output</li>
 *   <li>{@link #CLEF} — a joint head that decides all the questions of a request in one prompt</li>
 *   <li>{@link #PPLX_DECIDER} — same as openjev, label codes of 1 or 2 letters</li>
 *   <li>{@link #VLLM_DECISION2} — a joint head over one question per prompt, the options marked
 *       in the batch</li>
 *   <li>{@link #VLLM_D3} — same as pplx-decider, with non-causal attention</li>
 *   <li>{@link #GRANITE_SWITCH} — same as pplx-decider, run by an adapter built into the model</li>
 *   <li>{@link #H2O_LIGHTNING} — same as openjev, the label texts read from the GGUF</li>
 *   <li>{@link #MINICPM_OCR_JEV} — one score per class, the pooled output of the last token</li>
 * </ul>
 * Each family has its decider: {@link LabelDecider} (openjev, lev, nimble, pplx-decider, vllm-d3,
 * granite-switch, h2o-lightning, and {@link #CUSTOM}), {@link KevDecider}, {@link LayaDecider},
 * {@link ClefDecider} (clef, vllm-decision2) and {@link PooledDecider} (openbmb-minicpm-ocr-jev);
 * {@link LlamaDecider#create} picks it from the metadata.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public enum DecisionType {
  OPENJEV("openjev"),
  LEV("lev"),
  KEV("kev"),
  NIMBLE("nimble"),
  LAYA("laya"),
  CLEF("clef"),
  PPLX_DECIDER("pplx-decider"),
  VLLM_DECISION2("vllm-decision2"),
  VLLM_D3("vllm-d3"),
  GRANITE_SWITCH("granite-switch"),
  H2O_LIGHTNING("h2o-lightning"),
  MINICPM_OCR_JEV("openbmb-minicpm-ocr-jev"),
  /**
   * Any causal model scored on caller-chosen label tokens ({@link LabelDecider#custom}); never read
   * from GGUF metadata.
   */
  CUSTOM(null);

  private final String metaName;

  DecisionType(String metaName) {
    this.metaName = metaName;
  }

  /** The value of {@code <arch>.decision.type} for this family, {@code null} for {@link #CUSTOM}. */
  public String metaName() {
    return metaName;
  }

  /**
   * Reads the decision type of a model from its GGUF metadata.
   *
   * @param arena Used to allocate the metadata C strings
   * @param model The model to inspect
   * @return The decision type, or empty if the model is not a decision model
   * @throws LlamaException if the model declares a decision family this library does not know
   */
  public static Optional<DecisionType> of(Arena arena, LlamaModel model) {
    String arch = model.metaVal(arena, "general.architecture");
    if (arch == null) {
      return Optional.empty();
    }
    String name = model.metaVal(arena, arch + ".decision.type");
    if (name == null) {
      return Optional.empty();
    }
    return Optional.of(fromMetaName(name));
  }

  static DecisionType fromMetaName(String name) {
    return Arrays.stream(values())
      .filter(type -> name.equals(type.metaName))
      .findFirst()
      .orElseThrow(() ->
        new LlamaException(
          "unsupported decision model type: %s".formatted(name)
        )
      );
  }
}
