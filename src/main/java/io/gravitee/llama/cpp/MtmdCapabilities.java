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

/**
 * What a model and its multimodal projector can read and produce ({@link MtmdContext#capabilities()}).
 *
 * @param chat                 The model can chat (false for speech-only and realtime transcription models)
 * @param vision               It reads images
 * @param audio                It reads audio
 * @param video                It reads video ({@link MtmdVideo}); also needs a build with video support
 * @param audioStream          It transcribes audio as it arrives
 * @param textGeneration       It generates text
 * @param audioGeneration      It generates speech
 * @param audioGenerationVoice It clones a voice from a reference clip
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record MtmdCapabilities(
  boolean chat,
  boolean vision,
  boolean audio,
  boolean video,
  boolean audioStream,
  boolean textGeneration,
  boolean audioGeneration,
  boolean audioGenerationVoice
) {}
