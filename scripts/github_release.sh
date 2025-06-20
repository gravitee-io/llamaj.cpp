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

cd $HOME_DIR/llamaj.cpp/

# Built release note
RELEASE_VERSION=$(mvn help:evaluate -Dexpression=project.version -q -DforceStdout)
echo "Release note: ${RELEASE_VERSION}"
cat ./release-note.md

# Push to repository
echo "Push release to branch: ${CIRCLE_BRANCH}"
git push -u origin ${CIRCLE_BRANCH}
git push --tags origin ${CIRCLE_BRANCH}
gh release create $RELEASE_VERSION -F ./release-note.md --title "v$RELEASE_VERSION" --target ${CIRCLE_BRANCH}