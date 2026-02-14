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
 * Performance metrics extracted from llama.cpp context and sampler.
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record LlamaPerformance(
  ContextPerformance context,
  SamplerPerformance sampler
) {
  /**
   * Context performance metrics from llama.cpp.
   */
  public record ContextPerformance(
    double startTimeMs,
    double loadTimeMs,
    double promptEvalTimeMs,
    double evalTimeMs,
    int promptTokensEvaluated,
    int tokensGenerated,
    int tokensReused
  ) {
    /**
     * Calculate prompt processing speed in tokens per second.
     * @return tokens per second, or 0 if no prompt evaluation occurred
     */
    public double promptTokensPerSecond() {
      if (promptEvalTimeMs > 0 && promptTokensEvaluated > 0) {
        return (promptTokensEvaluated / promptEvalTimeMs) * 1000.0;
      }
      return 0.0;
    }

    /**
     * Calculate token generation speed in tokens per second.
     * @return tokens per second, or 0 if no tokens were generated
     */
    public double generationTokensPerSecond() {
      if (evalTimeMs > 0 && tokensGenerated > 0) {
        return (tokensGenerated / evalTimeMs) * 1000.0;
      }
      return 0.0;
    }

    /**
     * Calculate total processing time in milliseconds.
     * @return total time (prompt eval + generation)
     */
    public double totalProcessingTimeMs() {
      return promptEvalTimeMs + evalTimeMs;
    }
  }

  /**
   * Sampler performance metrics from llama.cpp.
   */
  public record SamplerPerformance(double samplingTimeMs, int sampleCount) {
    /**
     * Calculate average time per sample in milliseconds.
     * @return average sampling time, or 0 if no samples
     */
    public double averageSamplingTimeMs() {
      if (sampleCount > 0) {
        return samplingTimeMs / sampleCount;
      }
      return 0.0;
    }
  }

  /**
   * Calculate prompt processing speed in tokens per second.
   * @return tokens per second from context
   */
  public double promptTokensPerSecond() {
    return context.promptTokensPerSecond();
  }

  /**
   * Calculate token generation speed in tokens per second.
   * @return tokens per second from context
   */
  public double generationTokensPerSecond() {
    return context.generationTokensPerSecond();
  }

  /**
   * Calculate total processing time in milliseconds.
   * @return total time (prompt eval + generation)
   */
  public double totalProcessingTimeMs() {
    return context.totalProcessingTimeMs();
  }

  /**
   * Calculate average time per sample in milliseconds.
   * @return average sampling time from sampler
   */
  public double averageSamplingTimeMs() {
    return sampler.averageSamplingTimeMs();
  }
}
