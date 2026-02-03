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

import static org.assertj.core.api.Assertions.assertThat;

import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * Tests for MtmdMedia interface and implementations.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
class MtmdMediaTest {

  @Test
  void should_create_mock_image_media() {
    // Create a mock implementation to test the interface
    MtmdMedia mockImage = new MtmdMedia() {
      private boolean freed = false;

      @Override
      public MemorySegment getMemorySegment() {
        return MemorySegment.NULL;
      }

      @Override
      public void free() {
        freed = true;
      }

      @Override
      public boolean isFree() {
        return freed;
      }
    };

    assertThat(mockImage.getMemorySegment()).isEqualTo(MemorySegment.NULL);
    assertThat(mockImage.isFree()).isFalse();

    mockImage.free();
    assertThat(mockImage.isFree()).isTrue();
  }

  @Test
  void should_verify_mtmd_image_implements_media() {
    // Verify that MtmdImage class implements MtmdMedia interface
    assertThat(MtmdImage.class.getInterfaces()).contains(MtmdMedia.class);
  }

  @Test
  void should_verify_mtmd_audio_implements_media() {
    // Verify that MtmdAudio class implements MtmdMedia interface
    assertThat(MtmdAudio.class.getInterfaces()).contains(MtmdMedia.class);
  }

  @Test
  void should_verify_freeable_interface() {
    // Verify that MtmdMedia extends Freeable
    assertThat(Freeable.class.isAssignableFrom(MtmdMedia.class)).isTrue();
  }

  @Test
  void should_allow_polymorphic_assignment() {
    // Test that both implementations can be used as MtmdMedia
    MtmdMedia mockMedia1 = new MtmdMedia() {
      @Override
      public MemorySegment getMemorySegment() {
        return MemorySegment.NULL;
      }

      @Override
      public void free() {}

      @Override
      public boolean isFree() {
        return false;
      }
    };

    MtmdMedia mockMedia2 = new MtmdMedia() {
      @Override
      public MemorySegment getMemorySegment() {
        return MemorySegment.NULL;
      }

      @Override
      public void free() {}

      @Override
      public boolean isFree() {
        return false;
      }
    };

    // Both can be assigned to MtmdMedia type
    java.util.List<MtmdMedia> mediaList = java.util.List.of(
      mockMedia1,
      mockMedia2
    );
    assertThat(mediaList).hasSize(2);
    assertThat(mediaList).allMatch(m -> m != null);
  }
}
