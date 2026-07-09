#!/usr/bin/env bash
#
# Copyright © 2015 The Gravitee team (http://gravitee.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Download every model listed in scripts/ci-models.txt in parallel.
#
# Files that already exist (e.g. restored from the CircleCI cache) are skipped,
# so this doubles as the cache-fill step: on a cold cache it fetches everything
# concurrently; on a warm cache it only fills gaps and re-derives reasoning.gguf.
#
# Env:
#   MODELS_ROOT  Base dir; destinations from the manifest are resolved under it
#                (required, e.g. "$HOME_DIR/llamaj.cpp").
#
set -euo pipefail

ROOT="${MODELS_ROOT:?MODELS_ROOT must be set}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/ci-models.txt"

download_one() {
  local repo="$1" file="$2" dest="$3"
  local out="$ROOT/$dest"
  if [[ -s "$out" ]]; then
    echo "cache hit, skipping: $dest"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  echo "downloading $repo/$file -> $dest"
  curl -fSL --retry 5 --retry-delay 3 --retry-connrefused \
    -o "$out.part" \
    "https://huggingface.co/$repo/resolve/main/$file"
  mv "$out.part" "$out" # atomic: a half-written file never looks like a cache hit
}

pids=()
while read -r repo file dest _; do
  [[ -z "${repo:-}" || "$repo" == \#* ]] && continue
  download_one "$repo" "$file" "$dest" &
  pids+=("$!")
done < "$MANIFEST"

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ $status -ne 0 ]]; then
  echo "ERROR: one or more model downloads failed" >&2
  exit 1
fi

# reasoning.gguf is just a copy of the base model (kept out of the manifest so it
# is not fetched twice).
cp "$ROOT/models/model.gguf" "$ROOT/models/reasoning.gguf"

echo "all models ready under $ROOT/models"
