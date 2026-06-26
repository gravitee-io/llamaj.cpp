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

import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.MODEL_TO_DOWNLOAD;
import static io.gravitee.llama.cpp.LlamaCppTest.REASONING_MODEL_PATH;
import static io.gravitee.llama.cpp.LlamaCppTest.REASONNING_MODEL_TO_DOWNLOAD;
import static io.gravitee.llama.cpp.LlamaCppTest.getModelPath;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Verifies speculative decoding via {@link ConversationState#setDraft} is lossless: it must
 * reproduce the target's plain greedy output — both when draft == target (same GGUF) and when
 * the draft is a smaller, different model from the same family (Qwen3-0.6B draft, Qwen3-1.7B
 * target), which also exercises the verify/correction and rejection-sampling paths.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@Tag("integration")
class SpeculativeDecodingTest extends LlamaCppTest {

  private static final String PROMPT = "The capital of France is";
  private static final int MAX_TOKENS = 24;

  // Draft/target pair from the same family (same tokenizer/vocab = 151936) for the realistic
  // (draft != target) case. The draft reuses the existing Qwen3-0.6B reasoning.gguf.
  static final String SPEC_TARGET_MODEL_TO_DOWNLOAD =
    "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf";
  static final String SPEC_TARGET_MODEL_PATH = "models/spec-target.gguf";

  private static Arena arena;

  @BeforeAll
  static void beforeAll() {
    arena = Arena.ofConfined();
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);
  }

  @AfterAll
  static void afterAll() {
    LlamaRuntime.llama_backend_free();
    arena.close();
    arena = null;
  }

  @Test
  void speculative_matches_plain_greedy() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));

    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var arCtx = track(new LlamaContext(arena, model, cp));
    var specCtx = track(new LlamaContext(arena, model, cp));
    var draftCtx = track(new LlamaContext(arena, model, cp));

    var vocab = new LlamaVocab(model);

    // Reference: plain greedy decoding.
    var arSampler = track(new LlamaSampler(arena).greedy());
    var arState = ConversationState.create(
      arena,
      arCtx,
      new LlamaTokenizer(vocab, arCtx),
      arSampler
    )
      .setMaxTokens(MAX_TOKENS)
      .initialize(PROMPT);
    String greedy = new DefaultLlamaIterator(arState)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    // Speculative (greedy): same model as draft (lossless ⇒ identical output).
    var specSampler = track(new LlamaSampler(arena).greedy());
    var specState = ConversationState.create(
      arena,
      specCtx,
      new LlamaTokenizer(vocab, specCtx),
      specSampler
    )
      .setMaxTokens(MAX_TOKENS)
      .setDraft(draftCtx, SpeculativeConfig.greedy(4))
      .initialize(PROMPT);
    String speculative = new DefaultLlamaIterator(specState)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    System.out.println("greedy     : " + greedy);
    System.out.println("speculative: " + speculative);

    assertThat(speculative).isEqualTo(greedy);
    // Identical draft & target ⇒ every drafted token accepted.
    assertThat(specState.acceptRate()).isEqualTo(1.0);
  }

  @Test
  void stochastic_accepts_everything_when_draft_equals_target() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var specCtx = track(new LlamaContext(arena, model, cp));
    var draftCtx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);

    // p == q (same model), so the rejection test min(1, p/q) accepts every drafted token.
    var sampler = track(new LlamaSampler(arena).greedy());
    var state = ConversationState.create(
      arena,
      specCtx,
      new LlamaTokenizer(vocab, specCtx),
      sampler
    )
      .setMaxTokens(MAX_TOKENS)
      .setDraft(draftCtx, new SpeculativeConfig(4, 0.8f, 40, 0.95f, 42))
      .initialize(PROMPT);

    String text = new DefaultLlamaIterator(state)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    System.out.println(
      "stochastic : " + text + " (accept=" + state.acceptRate() + ")"
    );
    assertThat(text).isNotBlank();
    assertThat(state.acceptRate()).isEqualTo(1.0);
  }

  @Test
  void fused_batch_speculative_matches_per_sequence_greedy() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena)
      .nCtx(512)
      .nBatch(512)
      .nUBatch(512)
      .nSeqMax(2);
    var batchCtx = track(new LlamaContext(arena, model, cp));
    var draftCtx = track(new LlamaContext(arena, model, cp));
    var refCtx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);

    String[] prompts = { "The capital of France is", "Water boils at" };

    // References: single-stream greedy per prompt.
    String[] refs = new String[2];
    for (int i = 0; i < 2; i++) {
      var refSampler = track(new LlamaSampler(arena).greedy());
      var refState = ConversationState.create(
        arena,
        refCtx,
        new LlamaTokenizer(vocab, refCtx),
        refSampler,
        0
      )
        .setMaxTokens(MAX_TOKENS)
        .initialize(prompts[i]);
      refs[i] = new DefaultLlamaIterator(refState)
        .stream()
        .map(LlamaOutput::content)
        .reduce("", (a, b) -> a + b);
      refCtx.clearCache();
    }

    // Fused batch speculative: two sequences, each with a draft (greedy ⇒ lossless).
    var batchIt = new BatchIterator(arena, batchCtx);
    var sb0 = new StringBuilder();
    var sb1 = new StringBuilder();
    try {
      for (int i = 0; i < 2; i++) {
        var sampler = track(new LlamaSampler(arena).greedy());
        batchIt.addState(
          ConversationState.create(
            arena,
            batchCtx,
            new LlamaTokenizer(vocab, batchCtx),
            sampler,
            i
          )
            .setMaxTokens(MAX_TOKENS)
            .setDraft(draftCtx, SpeculativeConfig.greedy(4))
            .initialize(prompts[i])
        );
      }
      batchIt
        .stream()
        .forEach(o -> (o.sequenceId() == 0 ? sb0 : sb1).append(o.content()));
    } finally {
      batchIt.free();
    }

    System.out.println("seq0 ref/spec: " + refs[0] + " || " + sb0);
    System.out.println("seq1 ref/spec: " + refs[1] + " || " + sb1);
    assertThat(sb0.toString()).isEqualTo(refs[0]);
    assertThat(sb1.toString()).isEqualTo(refs[1]);
  }

  @Test
  void greedy_speculative_with_smaller_draft_matches_target_greedy() {
    Path targetPath = getModelPath(
      SPEC_TARGET_MODEL_PATH,
      SPEC_TARGET_MODEL_TO_DOWNLOAD
    );
    Path draftPath = getModelPath(
      REASONING_MODEL_PATH,
      REASONNING_MODEL_TO_DOWNLOAD
    );
    var targetModel = track(
      new LlamaModel(arena, targetPath, new LlamaModelParams(arena))
    );
    var draftModel = track(
      new LlamaModel(arena, draftPath, new LlamaModelParams(arena))
    );

    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var arCtx = track(new LlamaContext(arena, targetModel, cp));
    var specCtx = track(new LlamaContext(arena, targetModel, cp));
    var draftCtx = track(new LlamaContext(arena, draftModel, cp));
    var vocab = new LlamaVocab(targetModel);

    // Reference: the TARGET's own plain greedy decoding.
    var arState = ConversationState.create(
      arena,
      arCtx,
      new LlamaTokenizer(vocab, arCtx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .initialize(PROMPT);
    String targetGreedy = new DefaultLlamaIterator(arState)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    // Speculative: a smaller, weaker draft (0.6B) proposing for the larger target (1.7B).
    var specState = ConversationState.create(
      arena,
      specCtx,
      new LlamaTokenizer(vocab, specCtx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .setDraft(draftCtx, SpeculativeConfig.greedy(4))
      .initialize(PROMPT);
    String speculative = new DefaultLlamaIterator(specState)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    System.out.println("target greedy: " + targetGreedy);
    System.out.println(
      "speculative  : " +
        speculative +
        " (accept=" +
        specState.acceptRate() +
        ")"
    );

    assertThat(speculative).contains("Paris", "London", "Washington");
    assertThat(targetGreedy).contains("Paris", "London", "Washington");
    assertThat(specState.acceptRate()).isBetween(0.0, 1.0);
  }

  @Test
  void greedy_adaptive_speculative_matches_target_greedy() {
    Path targetPath = getModelPath(
      SPEC_TARGET_MODEL_PATH,
      SPEC_TARGET_MODEL_TO_DOWNLOAD
    );
    Path draftPath = getModelPath(
      REASONING_MODEL_PATH,
      REASONNING_MODEL_TO_DOWNLOAD
    );
    var targetModel = track(
      new LlamaModel(arena, targetPath, new LlamaModelParams(arena))
    );
    var draftModel = track(
      new LlamaModel(arena, draftPath, new LlamaModelParams(arena))
    );

    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var arCtx = track(new LlamaContext(arena, targetModel, cp));
    var specCtx = track(new LlamaContext(arena, targetModel, cp));
    var draftCtx = track(new LlamaContext(arena, draftModel, cp));
    var vocab = new LlamaVocab(targetModel);

    // Reference: the TARGET's own plain greedy decoding.
    var arState = ConversationState.create(
      arena,
      arCtx,
      new LlamaTokenizer(vocab, arCtx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .initialize(PROMPT);
    String targetGreedy = new DefaultLlamaIterator(arState)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    // Adaptive greedy: draft up to 8 tokens but stop early once draft confidence < 0.5. Lossless
    // w.r.t. the target's greedy output regardless of how many tokens each round speculates.
    var specState = ConversationState.create(
      arena,
      specCtx,
      new LlamaTokenizer(vocab, specCtx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .setDraft(draftCtx, SpeculativeConfig.greedyAdaptive(8, 1, 0.5f))
      .initialize(PROMPT);
    String speculative = new DefaultLlamaIterator(specState)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    System.out.println("target greedy  : " + targetGreedy);
    System.out.println(
      "adaptive spec  : " +
        speculative +
        " (accept=" +
        specState.acceptRate() +
        ")"
    );

    assertThat(speculative).contains("Paris", "London", "Washington");
    assertThat(targetGreedy).contains("Paris", "London", "Washington");
    assertThat(specState.acceptRate()).isBetween(0.0, 1.0);
  }

  @Test
  void stochastic_speculative_with_smaller_draft_runs() {
    Path targetPath = getModelPath(
      SPEC_TARGET_MODEL_PATH,
      SPEC_TARGET_MODEL_TO_DOWNLOAD
    );
    Path draftPath = getModelPath(
      REASONING_MODEL_PATH,
      REASONNING_MODEL_TO_DOWNLOAD
    );
    var targetModel = track(
      new LlamaModel(arena, targetPath, new LlamaModelParams(arena))
    );
    var draftModel = track(
      new LlamaModel(arena, draftPath, new LlamaModelParams(arena))
    );

    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var specCtx = track(new LlamaContext(arena, targetModel, cp));
    var draftCtx = track(new LlamaContext(arena, draftModel, cp));
    var vocab = new LlamaVocab(targetModel);

    // draft != target with sampling: exercises the rejection-sampling + residual draw path
    // against real logits (can't assert exact output — it's stochastic).
    var state = ConversationState.create(
      arena,
      specCtx,
      new LlamaTokenizer(vocab, specCtx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .setDraft(draftCtx, new SpeculativeConfig(4, 0.8f, 40, 0.95f, 42))
      .initialize(PROMPT);
    String text = new DefaultLlamaIterator(state)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    System.out.println(
      "stochastic (draft!=target): " +
        text +
        " (accept=" +
        state.acceptRate() +
        ")"
    );
    assertThat(text).isNotBlank();
    assertThat(state.acceptRate()).isBetween(0.0, 1.0);
  }

  @Test
  void speculative_stream_abandoned_via_try_with_resources_keeps_context_usable() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var specCtx = track(new LlamaContext(arena, model, cp));
    var draftCtx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);

    // Abandon a speculative stream after a few tokens. close() (try-with-resources) must free the
    // persistent native scratch and clear the sequence without throwing — onFinished never fires.
    int consumed;
    try (
      var it = new DefaultLlamaIterator(
        ConversationState.create(
          arena,
          specCtx,
          new LlamaTokenizer(vocab, specCtx),
          track(new LlamaSampler(arena).greedy())
        )
          .setMaxTokens(MAX_TOKENS)
          .setDraft(draftCtx, SpeculativeConfig.greedy(4))
          .initialize(PROMPT)
      )
    ) {
      consumed = it.stream().limit(3).map(LlamaOutput::content).toList().size();
    }
    assertThat(consumed).isLessThanOrEqualTo(3);

    // The contexts must remain usable for a fresh speculative generation afterwards: close() cleared
    // the KV, and the new state's Speculation builds its own scratch.
    var reuseState = ConversationState.create(
      arena,
      specCtx,
      new LlamaTokenizer(vocab, specCtx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .setDraft(draftCtx, SpeculativeConfig.greedy(4))
      .initialize(PROMPT);
    String text;
    try (var it = new DefaultLlamaIterator(reuseState)) {
      text = it
        .stream()
        .map(LlamaOutput::content)
        .reduce("", (a, b) -> a + b);
    }
    assertThat(text).isNotBlank();
  }

  @Test
  void ngram_greedy_matches_plain_greedy() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var arCtx = track(new LlamaContext(arena, model, cp));
    var ngCtx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);

    // Reference: plain greedy decoding.
    var arState = ConversationState.create(
      arena,
      arCtx,
      new LlamaTokenizer(vocab, arCtx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .initialize(PROMPT);
    String greedy = new DefaultLlamaIterator(arState)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    // N-gram (prompt-lookup) greedy drafting: no draft model. Lossless w.r.t. greedy decoding
    // regardless of how many proposed tokens the target accepts.
    var ngState = ConversationState.create(
      arena,
      ngCtx,
      new LlamaTokenizer(vocab, ngCtx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .setNgram(SpeculativeConfig.ngramGreedy(4, 2))
      .initialize(PROMPT);
    String ngram = new DefaultLlamaIterator(ngState)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    System.out.println("greedy: " + greedy);
    System.out.println(
      "ngram : " + ngram + " (accept=" + ngState.acceptRate() + ")"
    );
    assertThat(ngram).isEqualTo(greedy);
    assertThat(ngState.acceptRate()).isBetween(0.0, 1.0);
  }

  @Test
  void fused_batch_ngram_matches_per_sequence_greedy() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena)
      .nCtx(512)
      .nBatch(512)
      .nUBatch(512)
      .nSeqMax(2);
    var batchCtx = track(new LlamaContext(arena, model, cp));
    var refCtx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);

    String[] prompts = { "The capital of France is", "Water boils at" };

    String[] refs = new String[2];
    for (int i = 0; i < 2; i++) {
      var refState = ConversationState.create(
        arena,
        refCtx,
        new LlamaTokenizer(vocab, refCtx),
        track(new LlamaSampler(arena).greedy()),
        0
      )
        .setMaxTokens(MAX_TOKENS)
        .initialize(prompts[i]);
      refs[i] = new DefaultLlamaIterator(refState)
        .stream()
        .map(LlamaOutput::content)
        .reduce("", (a, b) -> a + b);
      refCtx.clearCache();
    }

    // Fused batch n-gram drafting: two sequences, each prompt-lookup (greedy ⇒ lossless).
    var batchIt = new BatchIterator(arena, batchCtx);
    var sb0 = new StringBuilder();
    var sb1 = new StringBuilder();
    try {
      for (int i = 0; i < 2; i++) {
        batchIt.addState(
          ConversationState.create(
            arena,
            batchCtx,
            new LlamaTokenizer(vocab, batchCtx),
            track(new LlamaSampler(arena).greedy()),
            i
          )
            .setMaxTokens(MAX_TOKENS)
            .setNgram(SpeculativeConfig.ngramGreedy(4, 2))
            .initialize(prompts[i])
        );
      }
      batchIt
        .stream()
        .forEach(o -> (o.sequenceId() == 0 ? sb0 : sb1).append(o.content()));
    } finally {
      batchIt.free();
    }

    assertThat(sb0.toString()).isEqualTo(refs[0]);
    assertThat(sb1.toString()).isEqualTo(refs[1]);
  }

  @Test
  void ngram_stochastic_runs() {
    Path path = getModelPath(MODEL_PATH, MODEL_TO_DOWNLOAD);
    var model = track(new LlamaModel(arena, path, new LlamaModelParams(arena)));
    var cp = new LlamaContextParams(arena).nCtx(512).nBatch(512).nUBatch(512);
    var ngCtx = track(new LlamaContext(arena, model, cp));
    var vocab = new LlamaVocab(model);

    var state = ConversationState.create(
      arena,
      ngCtx,
      new LlamaTokenizer(vocab, ngCtx),
      track(new LlamaSampler(arena).greedy())
    )
      .setMaxTokens(MAX_TOKENS)
      .setNgram(SpeculativeConfig.ngram(4, 2, 0.8f, 40, 0.95f, 42))
      .initialize(PROMPT);
    String text = new DefaultLlamaIterator(state)
      .stream()
      .map(LlamaOutput::content)
      .reduce("", (a, b) -> a + b);

    System.out.println(
      "ngram stochastic: " + text + " (accept=" + state.acceptRate() + ")"
    );
    assertThat(text).isNotBlank();
    assertThat(state.acceptRate()).isBetween(0.0, 1.0);
  }
}
