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
package io.gravitee.llama.cpp.modules;

import static io.gravitee.llama.cpp.GenerationState.TOOLS;
import static io.gravitee.llama.cpp.modules.StateEvaluation.*;
import static java.util.function.Function.identity;
import static java.util.function.Predicate.not;
import static java.util.stream.Collectors.toMap;

import io.gravitee.llama.cpp.GenerationState;
import io.gravitee.llama.cpp.StateBounds;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * State-machine over generation channels (ANSWER / REASONING / TOOLS) driven by tag markers.
 *
 * <p>Two matching entry points:
 * <ul>
 *   <li>{@link #evaluate(Context)} — the historical single-piece string-equality matching (a
 *   marker matches only when one detokenized piece equals it exactly).</li>
 *   <li>{@link #evaluateToken} — TEXT-based streaming matching, robust to arbitrary
 *   tokenizations of the same marker text (fused pieces, subword splits, or pieces spanning
 *   the marker boundary such as {@code "thought\nThe"}). Pieces whose accumulated text is a
 *   strict prefix of a candidate marker are buffered (emitted as an empty delta) until the
 *   marker text is fully covered — <b>confirmation</b>: the state flips, the marker text is
 *   pure syntax and is suppressed (never emitted to any channel; its tokens are counted in
 *   the post-flip state), and the remainder of a boundary-spanning piece is emitted as the
 *   first text of the post-flip channel — or until the text diverges from every candidate —
 *   <b>refutation</b>: the buffered text is emitted in the current channel and the refuting
 *   piece is re-scanned (it may itself start a marker prefix).</li>
 * </ul>
 *
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class StateEvaluation
  implements Initializable<Config>, Evaluable<Context, GenerationState> {

  private Map<GenerationState, StateBounds> states;
  private Map<GenerationState, Boolean> occurredState;

  // Streaming text buffer: while pendingTokens > 0, `pending` holds the last pendingTokens
  // pieces' text, all of it being a strict prefix of at least one candidate marker.
  private final StringBuilder pending = new StringBuilder();
  private int pendingTokens;

  /**
   * A marker that has matched completely while a LONGER candidate sharing its
   * prefix is still possible — {@code <|call|>} is complete while
   * {@code <|call|><|start|>assistant…} may still be arriving. Held rather than
   * acted on, and settled when the stream diverges or ends.
   */
  private String provisionalMarker;

  private GenerationState provisionalTarget;

  @Override
  public boolean isInitialized() {
    return states != null && !states.isEmpty();
  }

  /** Piece-mode evaluation (single-piece equality). Kept for the historical contract. */
  @Override
  public GenerationState evaluate(Context context) {
    return isInitialized()
      ? switch (context.currentState) {
        case ANSWER -> detectNewState(context.piece);
        case REASONING, TOOLS -> detectEndState(
          context.currentState,
          context.piece
        );
        case null -> GenerationState.ANSWER;
      }
      : GenerationState.ANSWER;
  }

  /**
   * Text-based streaming evaluation of one generated token.
   *
   * <p>Returns the resulting state plus the text to emit for this step: normally the piece
   * itself ({@code emitTokens == 1}); an empty string while a marker prefix is buffered
   * ({@code emitTokens == 0}); on confirmation the marker text is suppressed and only the
   * boundary-spanning remainder (possibly empty) is emitted, with all covered tokens counted
   * in the post-flip state; on refutation the buffered text (plus this piece, unless it
   * restarts a marker prefix) is emitted in the current channel.
   *
   * @param tokenId the sampled token id (unused by the text matcher; kept for call-site
   *                symmetry)
   */
  public Emission evaluateToken(
    GenerationState currentState,
    int tokenId,
    String piece
  ) {
    if (!isInitialized() || currentState == null) {
      return new Emission(GenerationState.ANSWER, piece, 1);
    }
    if (piece == null) {
      piece = "";
    }
    if (pendingTokens == 0 && piece.isEmpty()) {
      // Nothing buffered and nothing to match (e.g. a partial-UTF-8 piece): plain emit.
      return new Emission(currentState, piece, 1);
    }
    return matchText(currentState, piece, true);
  }

  /**
   * Flushes any buffered marker-prefix text (generation ended while a candidate marker was
   * still unconfirmed). The flushed text belongs to the current channel.
   */
  public Emission flushPending(GenerationState currentState) {
    if (pendingTokens == 0) {
      return new Emission(currentState, "", 0);
    }
    if (provisionalMarker != null) {
      // Generation ended while a longer close marker was still possible — the agent flow,
      // where <|call|> IS the last token. Settle for the match that did complete so the
      // span closes and the turn does not end reported as still inside it.
      String marker = provisionalMarker;
      GenerationState target = provisionalTarget;
      if (currentState != GenerationState.ANSWER) {
        setAlreadyOccurredIfNecessary(currentState);
      }
      String remainder = pending.toString().substring(marker.length());
      int settled = pendingTokens;
      resetBuffer();
      return new Emission(target, remainder, settled);
    }
    String text = pending.toString();
    int n = pendingTokens;
    resetBuffer();
    return new Emission(currentState, text, n);
  }

  public boolean hasPending() {
    return pendingTokens > 0;
  }

  /* ----- text-based streaming matching ----- */

  /**
   * Matches the accumulated text (buffer + piece) against the current state's candidate
   * markers (open markers of not-yet-occurred states in ANSWER; the close marker of the
   * current state in REASONING/TOOLS). {@code allowRestart} guards the single-level
   * refutation re-scan.
   */
  private Emission matchText(
    GenerationState currentState,
    String piece,
    boolean allowRestart
  ) {
    if (currentState != GenerationState.ANSWER) {
      StateBounds bounds = states.get(currentState);
      if (stateAlreadyOccurred(bounds)) {
        // Legacy contract: a state whose close already occurred resolves to ANSWER.
        return new Emission(GenerationState.ANSWER, piece, 1);
      }
    }

    String accumulated = pendingTokens == 0
      ? piece
      : pending.toString() + piece;

    // Candidate set — channels CHAIN rather than nest (Harmony-style): in ANY state the
    // candidates are (a) the current state's close marker (→ ANSWER) and (b) the OPEN
    // markers of the OTHER states (→ that state directly, the current span implicitly
    // closing). Longest-match-wins across all candidates, since markers may share prefixes
    // (e.g. both Harmony continuations start with "<|end|><|start|>assistant<|channel|>").
    String bestMarker = null;
    GenerationState bestTarget = null;
    boolean anyPrefix = false;

    // (a) close markers.
    //
    // In ANSWER these are EVERY state's closes, not none. A close marker reaching the
    // answer channel is stray syntax by definition — the span it would have ended is not
    // open — and models do emit them there: after a tool call, generation restarts fresh
    // in ANSWER and Harmony still prefixes its reply with the final-channel header, so
    // without this the header is unmatchable and lands in the user's content as
    // "<|channel|>final<|message|>DONE". Matching only suppresses; the state is already
    // ANSWER, so nothing else changes.
    for (StateBounds bounds : closeCandidates(currentState)) {
      for (String marker : bounds.ends()) {
        if (marker == null || marker.isBlank()) {
          continue;
        }
        if (accumulated.startsWith(marker)) {
          if (bestMarker == null || marker.length() > bestMarker.length()) {
            bestMarker = marker;
            bestTarget = GenerationState.ANSWER;
          }
        } else if (marker.startsWith(accumulated)) {
          anyPrefix = true;
        }
      }
    }

    // (b) the other states' open markers (cross-transitions when not in ANSWER)
    for (Entry<GenerationState, StateBounds> e : states.entrySet()) {
      StateBounds bounds = e.getValue();
      // A channel's OWN openers count while inside it, when it repeats. Models
      // re-announce the channel they are already in — Harmony emits a second
      // <|channel|>analysis<|message|> mid-thought — and skipping it leaves the
      // header in the reasoning text as raw protocol. The transition is a no-op;
      // the point is that the marker is recognised, and therefore suppressed.
      boolean ownChannel = bounds.state() == currentState;
      if (
        (ownChannel && !bounds.repeatable()) ||
        stateAlreadyOccurred(bounds) ||
        isNullOrBlank(bounds)
      ) {
        continue;
      }
      // A channel may have several opening markers (Harmony: commentary and analysis).
      for (String marker : bounds.starts()) {
        if (marker == null || marker.isBlank()) {
          continue;
        }
        if (accumulated.startsWith(marker)) {
          if (bestMarker == null || marker.length() > bestMarker.length()) {
            bestMarker = marker;
            bestTarget = bounds.state();
          }
        } else if (marker.startsWith(accumulated)) {
          anyPrefix = true;
        }
      }
    }

    if (bestMarker != null && anyPrefix) {
      // Matched, but a LONGER candidate is still viable. Committing now makes the longer
      // marker unreachable forever; waiting without remembering this match strands the
      // state machine when it never arrives. Hold it, and settle on divergence or at the
      // end of the stream.
      provisionalMarker = bestMarker;
      provisionalTarget = bestTarget;
      pending.append(piece);
      pendingTokens++;
      return new Emission(currentState, "", 0);
    }

    if (bestMarker != null) {
      // Confirmed: marker text suppressed; the boundary-spanning remainder (if any) is the
      // first text of the post-flip channel; all covered tokens are counted post-flip.
      if (currentState != GenerationState.ANSWER) {
        setAlreadyOccurredIfNecessary(currentState);
      }
      String remainder = accumulated.substring(bestMarker.length());
      int n = pendingTokens + 1;
      resetBuffer();
      return new Emission(bestTarget, remainder, n);
    }

    if (anyPrefix) {
      // Still a strict prefix of at least one candidate marker: buffer, emit nothing.
      pending.append(piece);
      pendingTokens++;
      return new Emission(currentState, "", 0);
    }

    if (pendingTokens == 0) {
      // No buffer, no match: plain content.
      return new Emission(currentState, piece, 1);
    }

    if (provisionalMarker != null) {
      // The longer candidate never came: settle for the match we held.
      String marker = provisionalMarker;
      GenerationState target = provisionalTarget;
      if (currentState != GenerationState.ANSWER) {
        setAlreadyOccurredIfNecessary(currentState);
      }
      String remainder = accumulated.substring(marker.length());
      int n = pendingTokens + 1;
      resetBuffer();
      return new Emission(target, remainder, n);
    }

    // Refutation: the accumulated text diverged from every candidate. Flush the buffered
    // text to the current channel and re-scan the refuting piece alone (it may itself start
    // a marker prefix).
    String flushed = pending.toString();
    int flushedTokens = pendingTokens;
    resetBuffer();
    if (!allowRestart) {
      return new Emission(currentState, flushed + piece, flushedTokens + 1);
    }
    Emission restart = matchText(currentState, piece, false);
    return new Emission(
      restart.state(),
      flushed + restart.emit(),
      flushedTokens + restart.emitTokens()
    );
  }

  /**
   * The states whose close markers are candidates right now: the current state when
   * inside a span, every configured state when in ANSWER — where a close marker can
   * only be leftover syntax.
   */
  private List<StateBounds> closeCandidates(GenerationState currentState) {
    if (currentState != GenerationState.ANSWER) {
      return List.of(states.get(currentState));
    }
    return List.copyOf(states.values());
  }

  private void resetBuffer() {
    pending.setLength(0);
    pendingTokens = 0;
    provisionalMarker = null;
    provisionalTarget = null;
  }

  /* ----- piece-mode internals (unchanged semantics) ----- */

  private GenerationState detectEndState(
    GenerationState currentState,
    String piece
  ) {
    var state = states.get(currentState);

    if (stateAlreadyOccurred(state)) {
      return GenerationState.ANSWER;
    }

    if (state.ends().contains(piece)) {
      setAlreadyOccurredIfNecessary(currentState);
      return GenerationState.ANSWER;
    }

    return currentState;
  }

  private void setAlreadyOccurredIfNecessary(GenerationState currentState) {
    var bounds = states.get(currentState);
    boolean closesForGood = bounds == null || !bounds.repeatable();
    this.occurredState.put(currentState, closesForGood);
  }

  private GenerationState detectNewState(String piece) {
    return states
      .entrySet()
      .stream()
      .filter(not(e -> this.stateAlreadyOccurred(e.getValue())))
      .map(Entry::getValue)
      .filter(not(StateEvaluation::isNullOrBlank))
      .filter(stateBounds -> stateBounds.starts().contains(piece))
      .map(StateBounds::state)
      .findFirst()
      .orElse(GenerationState.ANSWER);
  }

  /**
   * Resolves the state generation should start in for the given prompt.
   *
   * <p>Chat templates may pre-fill a state's open tag at the end of the prompt (e.g. a template
   * ending with {@code <think>}), and continuation prompts may end anywhere inside an
   * unfinished span (open tag present, close tag not yet emitted). In both cases the model
   * never (re-)emits the open tag itself and generation must begin inside that state, so a
   * prompt whose LAST open-tag occurrence is not followed by the corresponding close tag seeds
   * that state. Matching is string-based on the rendered prompt, so it is independent of how
   * the markers tokenize.
   */
  public GenerationState initialState(String prompt) {
    if (!isInitialized() || prompt == null) {
      return GenerationState.ANSWER;
    }
    var trimmed = prompt.stripTrailing();
    return states
      .values()
      .stream()
      .filter(not(StateEvaluation::isNullOrBlank))
      .filter(stateBounds -> insideOpenSpan(trimmed, stateBounds))
      .map(StateBounds::state)
      .findFirst()
      .orElse(GenerationState.ANSWER);
  }

  private static boolean insideOpenSpan(String prompt, StateBounds bounds) {
    // Any opening marker may be the unclosed one.
    int lastStart = -1;
    for (String marker : bounds.starts()) {
      if (marker != null && !marker.isBlank()) {
        lastStart = Math.max(lastStart, prompt.lastIndexOf(marker));
      }
    }
    if (lastStart < 0) {
      return false;
    }
    int lastEnd = -1;
    for (String end : bounds.ends()) {
      if (end != null && !end.isBlank()) {
        lastEnd = Math.max(lastEnd, prompt.lastIndexOf(end));
      }
    }
    return lastStart > lastEnd;
  }

  private static boolean isNullOrBlank(StateBounds stateBounds) {
    return stateBounds
      .starts()
      .stream()
      .allMatch(m -> m == null || m.isBlank());
  }

  private Boolean stateAlreadyOccurred(StateBounds stateBounds) {
    if (stateBounds == null) {
      return true;
    }

    if (stateBounds.repeatable()) {
      return false;
    }

    return occurredState.get(stateBounds.state());
  }

  @Override
  public void initialize(Config config) {
    this.states = config.states
      .stream()
      .collect(toMap(StateBounds::state, identity()));
    this.occurredState = this.states.keySet()
      .stream()
      .collect(toMap(identity(), __ -> false));
    resetBuffer();
  }

  public record Config(List<StateBounds> states) {}

  public record Context(GenerationState currentState, String piece) {}

  /**
   * Result of evaluating one token: the resulting generation state, the text to emit for this
   * step, and how many generated tokens it covers (0 while buffering a marker prefix; N when
   * buffered pieces resolve — marker text itself is always suppressed). Emitted text always
   * belongs to {@code state}.
   */
  public record Emission(GenerationState state, String emit, int emitTokens) {}
}
