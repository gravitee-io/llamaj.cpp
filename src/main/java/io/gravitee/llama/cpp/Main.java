package io.gravitee.llama.cpp;

import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;

import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import static io.gravitee.llama.cpp.LlamaRuntime.*;

public class Main {

    public static final Arena ARENA = Arena.ofAuto();

    public static void main(String[] args) {
        String modelGguf = args[0];
        String systemMessage = args[1];

        LlamaLibLoader.load();
        ggml_backend_load_all();

        var logger = new LlamaLogger(ARENA);
        logger.setLogging(LlamaLogLevel.ERROR);

        var modelParameters = new LlamaModelParams(ARENA);
        modelParameters.nGpuLayers(999).useMlock(true).useMmap(true);

        LlamaModel model = new LlamaModel(ARENA, Path.of(modelGguf).toAbsolutePath(), modelParameters);
        LlamaVocab vocab = new LlamaVocab(model);

        LlamaSampler sampler = new LlamaSampler(ARENA)
                .temperature(0.4f)
                .minP(0.1f, 40)
                .topK(10)
                .topP(0.2f, 10)
                .seed(new Random().nextInt());

        var contextParams = new LlamaContextParams(ARENA)
                .nCtx(512)
                .nBatch(512)
                .nSeqMax(256);

        String input = "";
        while (!input.trim().equals("bye")) {
            Scanner scanIn = new Scanner(System.in);
            System.out.print("Please enter your prompt: ");
            input = scanIn.nextLine();

            if (input.isBlank()) {
                break;
            }

            String prompt = buildPrompt(model, systemMessage, input, contextParams);

            var it = new LlamaIterator(ARENA, model, contextParams, vocab, sampler)
                    .setQuota(256)
                    .initialize(prompt);

            while (it.hasNext()) {
                System.out.print(it.next().content());
            }

            it.close();

            System.out.println();
        }

        sampler.free();
        model.free();
    }

    private static String buildPrompt(LlamaModel model, String systemMessage, String input, LlamaContextParams contextParams) {
        try (Arena arena = Arena.ofConfined()) {
            LlamaTemplate llamaTemplate = new LlamaTemplate(model);
            var messages = new LlamaChatMessages(arena, List.of(
                    new LlamaChatMessage(arena, Role.SYSTEM, systemMessage),
                    new LlamaChatMessage(arena, Role.USER, input)
            ));
            return llamaTemplate.applyTemplate(arena, messages, contextParams.nCtx());
        }
    }
}
