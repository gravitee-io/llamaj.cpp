package io.gravitee.llama.cpp;

import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import static io.gravitee.llama.cpp.llama_h_1.*;


public class Main {

    public static final Arena ARENA = Arena.ofAuto();

    public static void main(String[] args) {
        String nativeLibPath = args[0];
        String modelGguf = args[1];

        System.load(Path.of(nativeLibPath).toAbsolutePath().toString());

        ggml_backend_load_all();

        var modelParameters = new LlamaModelParams(ARENA);
        modelParameters.nGpuLayers(99).nGpuLayers();

        LlamaModel model = new LlamaModel(ARENA, Path.of(modelGguf).toAbsolutePath(), modelParameters);

        var contextParams = new LlamaContextParams(ARENA)
                .nCtx(512)
                .nBatch(512);

        LlamaContext context = new LlamaContext(model, contextParams);

        LlamaSampler sampler = new LlamaSampler(ARENA)
                .minP(0.05f)
                .temperature(0.8f)
                .seed(new Random().nextInt());

        LlamaVocab vocab = new LlamaVocab(model);

        String input = "";
        while (!input.trim().equals("bye")) {
            Scanner scanIn = new Scanner(System.in);
            System.out.print("Please enter your prompt: ");
            input = scanIn.nextLine();

            if (input.isBlank()) {
                break;
            }

            String prompt;
            try (Arena arena = Arena.ofConfined()) {
                LlamaTemplate llamaTemplate = new LlamaTemplate(model);
                var messages = new LlamaChatMessages(arena, List.of(
                        new LlamaChatMessage(arena, Role.SYSTEM, """
                                You are Yoda, a powerful Jedi Master with the knowledge of the universe.
                                Answer the question to the best of your ability.
                                """
                        ),
                        new LlamaChatMessage(arena, Role.USER, input)
                ));
                prompt = llamaTemplate.applyTemplate(arena, messages, contextParams.nCtx());
            }

            var llamaIterator = new LlamaIterator(
                    context,
                    vocab,
                    sampler,
                    prompt,
                    contextParams.nCtx()
            );

            for (LlamaIterator it = llamaIterator; it.hasNext(); ) {
                System.out.print(it.next());
            }

            llamaIterator.close();
            System.out.println();
        }

        llama_sampler_free(sampler.segment);
        llama_free(context.segment);
        llama_model_free(model.segment);
    }
}
