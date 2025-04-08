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

cd "$HOME_DIR/gravitee-llama-cpp/"
OLD_VERSION=$(mvn help:evaluate -Dexpression=project.version -q -DforceStdout)

RECOMMENDED_BUMP=$(conventional-recommended-bump -p conventionalcommits --quiet)
RELEASE_VERSION=$(semver $OLD_VERSION -i $RECOMMENDED_BUMP)

echo "release version: $RELEASE_VERSION"

mvn versions:set -DgenerateBackupPoms=false -DnewVersion="$RELEASE_VERSION"

conventional-changelog -p conventionalcommits -i CHANGELOG.md -s
tail -n+3 CHANGELOG.md | sed -r "s/###/##/g" > ./release-note.md
rm CHANGELOG.md

git add pom.xml
git commit -m "chore: release $RELEASE_VERSION [skip ci]"
git tag -a "$RELEASE_VERSION" -m "$RELEASE_VERSION"