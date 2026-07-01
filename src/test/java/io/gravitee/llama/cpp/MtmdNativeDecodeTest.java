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
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Regression test for the NATIVE media-decode path.
 *
 * <p>{@link MtmdImage#fromBytesNative} decodes raw file bytes with the native
 * stb_image decoder via {@code mtmd_helper_bitmap_init_from_buf}. This is the
 * path production callers (e.g. gravitee-ai-server) use, and it is distinct from
 * {@link MtmdImage#fromFile}/{@link MtmdImage#fromBytes}, which decode in Java
 * and call {@code mtmd_bitmap_init}. The other multimodal tests only exercise the
 * Java-decode path, so a signature drift between the hand-written
 * production. This test guards the native entry points explicitly.
 */
class MtmdNativeDecodeTest extends LlamaCppTest {

  private static Arena arena = Arena.ofShared();

  @BeforeAll
  public static void beforeAll() {
    String libPath = LlamaLibLoader.load();
    LlamaRuntime.llama_backend_init();
    LlamaRuntime.ggml_backend_load_all_from_path(arena, libPath);
    System.out.println("Libraries loaded at: " + libPath);
    System.out.println("Devices registered: " + ggml_backend_reg_count());
  }

  @Test
  void fromBytesNative_should_decode_image_via_native_helper()
    throws IOException, URISyntaxException {
    var mtmdContext = loadVlContext();
    assertThat(mtmdContext.supportsVision()).isTrue();

    byte[] imageBytes = Files.readAllBytes(
      Path.of(getClass().getClassLoader().getResource("man.jpg").toURI())
    );

    var nativeImage = MtmdImage.fromBytesNative(arena, mtmdContext, imageBytes);
    try {
      assertThat(nativeImage.getMemorySegment().address()).isNotZero();
    } finally {
      nativeImage.free();
    }

    // Java decode path (mtmd_bitmap_init) must keep working too.
    var javaImage = MtmdImage.fromBytes(arena, imageBytes);
    try {
      assertThat(javaImage.getMemorySegment().address()).isNotZero();
    } finally {
      javaImage.free();
    }
  }

  private MtmdContext loadVlContext() throws IOException {
    Path mainModel = getModelPath(MODEL_VL_PATH, VL_TEXT);
    if (!Files.isRegularFile(mainModel)) {
      throw new IOException("Main VL model not found: " + mainModel);
    }
    var llamaModel = track(
      new LlamaModel(arena, mainModel, new LlamaModelParams(arena))
    );

    Path mmproj = getModelPath(VL_MMPROJ_PATH, VL_MMPROJ);
    if (!Files.isRegularFile(mmproj)) {
      throw new IOException("VL mmproj not found: " + mmproj);
    }
    var params = new MtmdContextParams(arena).useGpu(true).mediaMarker("<IMG>");
    return track(
      new MtmdContext(arena, llamaModel, mmproj.toAbsolutePath(), params)
    );
  }
}
