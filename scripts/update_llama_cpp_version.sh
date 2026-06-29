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

set -euo pipefail

# --- Configuration ---
REPO_URL="https://github.com/ggml-org/llama.cpp"
CLONE_DIR="$HOME_DIR/llama.cpp"
PROJECT_DIR="$HOME_DIR/llamaj.cpp"
POM_FILE="$PROJECT_DIR/pom.xml"
CIRCLECI_CONFIG="$PROJECT_DIR/.circleci/config.yml"
README_FILE="$PROJECT_DIR/README.md"

# --- Clone llama.cpp repo ---
echo "Cloning llama.cpp into $CLONE_DIR..."
git clone --single-branch "$REPO_URL" "$CLONE_DIR"

# --- Get latest llama.cpp version ---
echo "Fetching latest llama.cpp version from GitHub..."
cd "$CLONE_DIR"
NEW_LLAMA_CPP_VERSION=$(gh release list --limit 1 --json tagName | jq -r '.[0].tagName')

# --- Get current version from project ---
cd "$PROJECT_DIR"
echo "Detecting current llama.cpp version in project..."
OLD_LLAMA_CPP_VERSION=$(grep -oP '<llama.cpp.version>\K(b[0-9]+)(?=</llama.cpp.version>)' "$POM_FILE")

# --- Create a branch for the update ---
branch_name="chore/llama.cpp-$OLD_LLAMA_CPP_VERSION-to-$NEW_LLAMA_CPP_VERSION"
echo "Creating branch $branch_name..."
git checkout -b "$branch_name"

# --- Update version in files ---
echo "Updating versions from $OLD_LLAMA_CPP_VERSION to $NEW_LLAMA_CPP_VERSION..."
sed -i'' -E "s/$OLD_LLAMA_CPP_VERSION/$NEW_LLAMA_CPP_VERSION/g" "$POM_FILE"
sed -i'' -E "s/$OLD_LLAMA_CPP_VERSION/$NEW_LLAMA_CPP_VERSION/g" "$CIRCLECI_CONFIG"
# README badge + attribution reference the pinned llama.cpp tag (e.g. b9673).
sed -i'' -E "s/$OLD_LLAMA_CPP_VERSION/$NEW_LLAMA_CPP_VERSION/g" "$README_FILE"

# --- Refresh the bundled llama.cpp license ---
# Keep licenses/LICENSE-llama-cpp in sync with the version we ship
LICENSE_DEST="$PROJECT_DIR/licenses/LICENSE-llama-cpp"
echo "Refreshing $LICENSE_DEST from llama.cpp $NEW_LLAMA_CPP_VERSION..."
mkdir -p "$PROJECT_DIR/licenses"
cp "$CLONE_DIR/LICENSE" "$LICENSE_DEST"

# --- Commit and push changes ---
echo "Committing and pushing changes..."
git add "$POM_FILE" "$CIRCLECI_CONFIG" "$README_FILE" "$LICENSE_DEST"

TITLE="feat(deps): update llama.cpp from $OLD_LLAMA_CPP_VERSION to $NEW_LLAMA_CPP_VERSION"
git commit -m "$TITLE"
git push origin "$branch_name"

# --- Create GitHub PR ---
echo "Creating pull request..."
gh pr create --title "$TITLE" \
  --body "This PR updates llama.cpp from $OLD_LLAMA_CPP_VERSION to $NEW_LLAMA_CPP_VERSION."
