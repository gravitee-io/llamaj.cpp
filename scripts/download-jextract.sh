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

# Default values
OS=""
PLATFORM=""
JEXTRACT_DIR="${PROJECT_BASEDIR:-.}/.jextract"

print_usage() {
  echo "Usage: $0 -o <os> -p <platform> [-d <jextract_dir>]"
  echo "Example: $0 -o macosx -p aarch64"
  echo "       $0 -o linux -p x86_64 -d /path/to/jextract"
}

# Parse command-line arguments
while getopts ":o:p:d:h" opt; do
  case ${opt} in
    o)
      OS=$OPTARG
      ;;
    p)
      PLATFORM=$OPTARG
      ;;
    d)
      JEXTRACT_DIR=$OPTARG
      ;;
    h)
      print_usage
      exit 0
      ;;
    \?)
      echo "❌ Invalid option: -$OPTARG" >&2
      print_usage
      exit 1
      ;;
    :)
      echo "❌ Option -$OPTARG requires an argument." >&2
      print_usage
      exit 1
      ;;
  esac
done

# Validate arguments
if [[ -z "$OS" || -z "$PLATFORM" ]]; then
  echo "❌ Missing required arguments."
  print_usage
  exit 1
fi

# Map OS and PLATFORM to download URLs
JEXTRACT_URL=""

# Define jextract download URLs based on OS and platform
# These URLs are for early access builds of jextract from OpenJDK
# See: https://jdk.java.net/jextract/

case "$OS" in
  macosx)
    case "$PLATFORM" in
      aarch64)
        JEXTRACT_URL="https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_macos-aarch64_bin.tar.gz"
        ;;
      *)
        echo "❌ Unsupported platform for macOS: $PLATFORM" >&2
        exit 1
        ;;
    esac
    ;;
  linux)
    case "$PLATFORM" in
      x86_64)
        JEXTRACT_URL="https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_linux-x64_bin.tar.gz"
        ;;
      *)
        echo "❌ Unsupported platform for Linux: $PLATFORM" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "❌ Unsupported OS: $OS" >&2
    exit 1
    ;;
esac

# Check if jextract binary already exists and is executable
JEXTRACT_BIN="${JEXTRACT_DIR}/bin/jextract"
if [[ -f "$JEXTRACT_BIN" && -x "$JEXTRACT_BIN" ]]; then
  echo "✅ jextract already exists at $JEXTRACT_BIN and is executable. Skipping download."
  exit 0
fi

# Create jextract directory if it doesn't exist
mkdir -p "$JEXTRACT_DIR"

# Create temporary directory for download
TMP_DIR="$(mktemp -d)"
ARCHIVE_NAME="jextract.tar.gz"

# Download jextract
echo "📥 Downloading jextract for $OS/$PLATFORM..."
echo "🔗 $JEXTRACT_URL"

curl -k -L -o "$TMP_DIR/$ARCHIVE_NAME" "$JEXTRACT_URL"

# Extract jextract
echo "📦 Extracting jextract to $JEXTRACT_DIR..."

# Extract directly into .jextract directory, flattening the structure
# This will extract all contents directly into .jextract/ without the jextract-25/ wrapper
tar -xzf "$TMP_DIR/$ARCHIVE_NAME" -C "$JEXTRACT_DIR" --strip-components=1

# Verify extraction - we should now have bin/jextract directly in .jextract/
if [[ ! -f "$JEXTRACT_BIN" ]]; then
  echo "❌ Failed to find jextract binary at $JEXTRACT_BIN after extraction." >&2
  rm -rf "$TMP_DIR"
  exit 1
fi

# Make sure jextract binary is executable
chmod +x "$JEXTRACT_BIN"

# Remove Windows-specific files that are unnecessary on Unix-like systems
rm -f "$JEXTRACT_DIR"/bin/jextract.bat
rm -f "$JEXTRACT_DIR"/bin/jextract.ps1

# Clean up empty directories
find "$JEXTRACT_DIR" -type d -empty -delete

# Clean up temp directory
rm -rf "$TMP_DIR"

# Verify jextract works
if "$JEXTRACT_BIN" --version >/dev/null 2>&1; then
  echo "✅ jextract downloaded and ready at $JEXTRACT_BIN"
  echo "   Version: $($JEXTRACT_BIN --version)"
else
  echo "❌ Failed to execute jextract." >&2
  exit 1
fi
