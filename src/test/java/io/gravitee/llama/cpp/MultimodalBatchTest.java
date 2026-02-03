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

import static io.gravitee.llama.cpp.LlamaRuntime.ggml_backend_reg_count;
import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests multimodal batch processing with the same image and multiple questions.
 */
class MultimodalBatchTest extends LlamaCppTest {

  private static Arena arena = Arena.ofShared();

  @BeforeAll
  public static void beforeAll() {
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);

    System.out.println("****************************");
    System.out.println("Libraries loaded at: " + libPath);
    System.out.println(
      "Number of devices registered: " + ggml_backend_reg_count()
    );
    System.out.println("****************************");
  }

  @Test
  void must_test_multimodal_batch_with_same_image()
    throws IOException, URISyntaxException {
    var logger = new LlamaLogger(arena);
    logger.setLogging(LlamaLogLevel.ERROR);

    // 1. Load the main Llama model
    var modelParameters = new LlamaModelParams(arena);
    Path mainModelAbsolutePath = getModelPath(MODEL_VL_PATH, VL_TEXT);
    if (
      !Files.exists(mainModelAbsolutePath) ||
      !Files.isRegularFile(mainModelAbsolutePath)
    ) {
      System.err.println(
        "Main model file does not exist or is not a regular file: " +
          mainModelAbsolutePath
      );
      throw new IOException(
        "Main model file not found: " + mainModelAbsolutePath
      );
    }
    System.out.println("Main model file exists: " + mainModelAbsolutePath);
    var llamaModel = new LlamaModel(
      arena,
      mainModelAbsolutePath,
      modelParameters
    );

    // 2. Setup MtmdContextParams and MtmdContext for multi-modal processing
    var mtmdContextParams = new MtmdContextParams(arena)
      .useGpu(true)
      .mediaMarker("<IMG>")
      .printTimings(true);

    Path mmprojModelAbsolutePath = getModelPath(VL_MMPROJ_PATH, VL_MMPROJ);
    if (
      !Files.exists(mmprojModelAbsolutePath) ||
      !Files.isRegularFile(mmprojModelAbsolutePath)
    ) {
      System.err.println(
        "MMProj model file does not exist or is not a regular file: " +
          mmprojModelAbsolutePath
      );
      throw new IOException(
        "MMProj model file not found: " + mmprojModelAbsolutePath
      );
    }
    System.out.println("MMProj model file exists: " + mmprojModelAbsolutePath);
    var mtmdContext = new MtmdContext(
      arena,
      llamaModel,
      mmprojModelAbsolutePath.toAbsolutePath(),
      mtmdContextParams
    );

    // 3. Create a regular LlamaContext for model inference
    var llamaContextParams = new LlamaContextParams(arena)
      .noPerf(false)
      .nCtx(8192)
      .nBatch(8192)
      .nSeqMax(3);
    var llamaContext = new LlamaContext(arena, llamaModel, llamaContextParams);

    // 4. Load the image
    Path imagePath = Path.of(
      getClass().getClassLoader().getResource("dog.jpg").toURI()
    );
    var mtmdImage = MtmdImage.fromFile(arena, imagePath);

    // 5. Prepare ConversationState with three different prompts
    var vocab = new LlamaVocab(llamaModel);
    var tokenizer = new LlamaTokenizer(vocab, llamaContext);

    // Three different questions about the same image using llama_chat_template
    String mediaMarker = mtmdContextParams.mediaMarker();
    var messages1 = new LlamaChatMessages(
      arena,
      List.of(
        new LlamaChatMessage(
          arena,
          Role.USER,
          mediaMarker + "\nWhat animals are in the picture?"
        )
      )
    );
    var messages2 = new LlamaChatMessages(
      arena,
      List.of(
        new LlamaChatMessage(
          arena,
          Role.USER,
          mediaMarker + "\nHow many animals are there in the picture?"
        )
      )
    );
    var messages3 = new LlamaChatMessages(
      arena,
      List.of(
        new LlamaChatMessage(
          arena,
          Role.USER,
          mediaMarker + "\nWhat's the main color of the picture?"
        )
      )
    );

    String[] prompts = {
      getPrompt(llamaModel, arena, messages1, llamaContextParams),
      getPrompt(llamaModel, arena, messages2, llamaContextParams),
      getPrompt(llamaModel, arena, messages3, llamaContextParams),
    };
    System.out.println("Formatted prompt 1: " + prompts[0]);
    System.out.println("Formatted prompt 2: " + prompts[1]);
    System.out.println("Formatted prompt 3: " + prompts[2]);

    // Create separate samplers for each state - samplers are stateful and must not be shared
    var sampler1 = new LlamaSampler(arena).greedy().seed(42);
    var sampler2 = new LlamaSampler(arena).greedy().seed(42);
    var sampler3 = new LlamaSampler(arena).greedy().seed(42);

    // Create three conversation states with unique sequence IDs
    var state1 = ConversationState.create(
      arena,
      llamaContext,
      tokenizer,
      sampler1,
      0
    )
      .initialize(prompts[0])
      .setImages(List.of(mtmdImage));

    var state2 = ConversationState.create(
      arena,
      llamaContext,
      tokenizer,
      sampler2,
      1
    )
      .initialize(prompts[1])
      .setImages(List.of(mtmdImage));

    var state3 = ConversationState.create(
      arena,
      llamaContext,
      tokenizer,
      sampler3,
      2
    )
      .initialize(prompts[2])
      .setImages(List.of(mtmdImage));

    // 6. Create batch iterator with all three states
    var batchIterator = new BatchIterator(arena, llamaContext, mtmdContext)
      .addState(state1)
      .addState(state2)
      .addState(state3);

    // Map to collect answers for each question
    var answers = new java.util.HashMap<Integer, StringBuilder>();
    answers.put(state1.getSequenceId(), new StringBuilder());
    answers.put(state2.getSequenceId(), new StringBuilder());
    answers.put(state3.getSequenceId(), new StringBuilder());

    // 7. Process all questions in parallel using next() pattern
    System.out.println("=== Multimodal Batch Processing ===");
    while (batchIterator.hasNext()) {
      var output = batchIterator.next();
      int seqId = output.sequenceId();
      answers.get(seqId).append(output.text());
    }

    // 8. Collect and display results
    System.out.println("\n=== Final Results ===");
    System.out.println(
      "Question 1 (What animals are in the picture?): " +
        answers.get(state1.getSequenceId())
    );
    System.out.println(
      "Question 2 (How many animals are there?): " +
        answers.get(state2.getSequenceId())
    );
    System.out.println(
      "Question 3 (What's the main color?): " +
        answers.get(state3.getSequenceId())
    );

    // 9. Verify all questions generated responses
    assertThat(state1.getAnswerTokens()).isGreaterThan(0);
    assertThat(state2.getAnswerTokens()).isGreaterThan(0);
    assertThat(state3.getAnswerTokens()).isGreaterThan(0);

    assertThat(state1.getFinishReason()).isIn(
      FinishReason.EOS,
      FinishReason.LENGTH,
      FinishReason.STOP
    );
    assertThat(state2.getFinishReason()).isIn(
      FinishReason.EOS,
      FinishReason.LENGTH,
      FinishReason.STOP
    );
    assertThat(state3.getFinishReason()).isIn(
      FinishReason.EOS,
      FinishReason.LENGTH,
      FinishReason.STOP
    );

    // 10. Free resources
    mtmdImage.free();
    mtmdContext.free();
    llamaContext.free();
    sampler1.free();
    sampler2.free();
    sampler3.free();
    llamaModel.free();
    batchIterator.free();
  }

  @Test
  void must_test_multimodal_batch_with_audio()
    throws IOException, URISyntaxException, javax.sound.sampled.UnsupportedAudioFileException {
    var logger = new LlamaLogger(arena);
    logger.setLogging(LlamaLogLevel.DEBUG);

    // 1. Load the Ultravox audio model
    var modelParameters = new LlamaModelParams(arena);
    Path mainModelAbsolutePath = getModelPath(
      AUDIO_MODEL_PATH,
      AUDIO_MODEL_TO_DOWNLOAD
    );
    if (
      !Files.exists(mainModelAbsolutePath) ||
      !Files.isRegularFile(mainModelAbsolutePath)
    ) {
      System.err.println(
        "Audio model file does not exist or is not a regular file: " +
          mainModelAbsolutePath
      );
      throw new IOException(
        "Audio model file not found: " + mainModelAbsolutePath
      );
    }
    System.out.println("Audio model file exists: " + mainModelAbsolutePath);
    var llamaModel = new LlamaModel(
      arena,
      mainModelAbsolutePath,
      modelParameters
    );

    // 2. Setup MtmdContextParams and MtmdContext for audio processing
    var mtmdContextParams = new MtmdContextParams(arena)
      .useGpu(true)
      .printTimings(true);

    Path mmprojModelAbsolutePath = getModelPath(
      AUDIO_MMPROJ_PATH,
      AUDIO_MMPROJ_TO_DOWNLOAD
    );
    if (
      !Files.exists(mmprojModelAbsolutePath) ||
      !Files.isRegularFile(mmprojModelAbsolutePath)
    ) {
      System.err.println(
        "Audio mmproj file does not exist or is not a regular file: " +
          mmprojModelAbsolutePath
      );
      throw new IOException(
        "Audio mmproj file not found: " + mmprojModelAbsolutePath
      );
    }
    System.out.println("Audio mmproj file exists: " + mmprojModelAbsolutePath);
    var mtmdContext = new MtmdContext(
      arena,
      llamaModel,
      mmprojModelAbsolutePath.toAbsolutePath(),
      mtmdContextParams
    );

    // 3. Create a regular LlamaContext for model inference
    var llamaContextParams = new LlamaContextParams(arena)
      .noPerf(false)
      .nCtx(8192)
      .nBatch(8192)
      .nSeqMax(2);
    var llamaContext = new LlamaContext(arena, llamaModel, llamaContextParams);

    // 4. Load audio file
    Path audioPath = Path.of(
      getClass().getClassLoader().getResource("test-2.wav").toURI()
    );
    var mtmdAudio = MtmdAudio.fromFile(
      arena,
      audioPath,
      mtmdContext.getAudioBitrate()
    );
    System.out.println(
      "Audio file loaded successfully with bitrate: " +
        mtmdContext.getAudioBitrate()
    );

    // 5. Prepare ConversationState with audio using Llama 3.2 chat template
    var vocab = new LlamaVocab(llamaModel);
    var tokenizer = new LlamaTokenizer(vocab, llamaContext);

    // Use the model's chat template with the default media marker
    String mediaMarker = mtmdContextParams.mediaMarker();

    var messages1 = new LlamaChatMessages(
      arena,
      List.of(
        new LlamaChatMessage(
          arena,
          Role.USER,
          mediaMarker + "\nWhat is in this audio?"
        )
      )
    );
    var messages2 = new LlamaChatMessages(
      arena,
      List.of(
        new LlamaChatMessage(
          arena,
          Role.USER,
          mediaMarker + "\nDescribe the audio content."
        )
      )
    );

    String[] prompts = {
      getPrompt(llamaModel, arena, messages1, llamaContextParams),
      getPrompt(llamaModel, arena, messages2, llamaContextParams),
    };
    System.out.println("Formatted prompt 1: " + prompts[0]);
    System.out.println("Formatted prompt 2: " + prompts[1]);

    // Create separate samplers for each state - samplers are stateful and must not be shared
    var audioSampler1 = new LlamaSampler(arena).greedy().seed(42);
    var audioSampler2 = new LlamaSampler(arena).greedy().seed(42);

    // Create two conversation states with audio
    var state1 = ConversationState.create(
      arena,
      llamaContext,
      tokenizer,
      audioSampler1,
      0
    )
      .initialize(prompts[0])
      .setMedia(List.of(mtmdAudio));

    var state2 = ConversationState.create(
      arena,
      llamaContext,
      tokenizer,
      audioSampler2,
      1
    )
      .initialize(prompts[1])
      .setMedia(List.of(mtmdAudio));

    // 6. Create batch iterator with audio states
    var batchIterator = new BatchIterator(arena, llamaContext, mtmdContext)
      .addState(state1)
      .addState(state2);

    // Map to collect answers
    var answers = new java.util.HashMap<Integer, StringBuilder>();
    answers.put(state1.getSequenceId(), new StringBuilder());
    answers.put(state2.getSequenceId(), new StringBuilder());

    // 7. Process all prompts in parallel
    System.out.println("=== Multimodal Audio Batch Processing ===");
    while (batchIterator.hasNext()) {
      var output = batchIterator.next();
      int seqId = output.sequenceId();
      answers.get(seqId).append(output.text());
    }

    // 8. Collect and display results
    System.out.println("\n=== Audio Processing Results ===");
    System.out.println(
      "Audio Question 1: " + answers.get(state1.getSequenceId())
    );
    System.out.println(
      "Audio Question 2: " + answers.get(state2.getSequenceId())
    );

    // 9. Verify all questions generated responses
    assertThat(state1.getAnswerTokens()).isGreaterThan(0);
    assertThat(state2.getAnswerTokens()).isGreaterThan(0);

    assertThat(state1.getFinishReason()).isIn(
      FinishReason.EOS,
      FinishReason.LENGTH,
      FinishReason.STOP
    );
    assertThat(state2.getFinishReason()).isIn(
      FinishReason.EOS,
      FinishReason.LENGTH,
      FinishReason.STOP
    );

    // 10. Free resources
    mtmdAudio.free();
    mtmdContext.free();
    llamaContext.free();
    audioSampler1.free();
    audioSampler2.free();
    llamaModel.free();
    batchIterator.free();
  }

  @AfterAll
  public static void afterAll() {
    arena = null;
    LlamaRuntime.llama_backend_free();
  }
}
