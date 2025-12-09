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
VERSION=""
DESTINATION=""

print_usage() {
  echo "Usage: $0 -o <os> -p <platform> -v <llama_cpp_version> -d <destination>"
  echo "Example: $0 -o macosx -p aarch64 -v 0.2.3 -d ./target/generated-sources"
}

# Parse command-line arguments
while getopts ":o:p:v:d:h" opt; do
  case ${opt} in
    o)
      OS=$OPTARG
      ;;
    p)
      PLATFORM=$OPTARG
      ;;
    v)
      VERSION=$OPTARG
      ;;
    d)
      DESTINATION=$OPTARG
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
if [[ -z "$OS" || -z "$PLATFORM" || -z "$VERSION" || -z "$DESTINATION" ]]; then
  echo "❌ Missing required arguments."
  print_usage
  exit 1
fi

# Debugging output to ensure correct arguments are received
echo "OS: $OS"
echo "Platform: $PLATFORM"
echo "Version: $VERSION"
echo "Destination: $DESTINATION"

if [ -d $DESTINATION/$OS/$PLATFORM ]; then
  rm -r $DESTINATION/$OS/$PLATFORM
fi

# Map OS and PLATFORM to the appropriate format for downloading
case "$OS" in
  macosx) OS_DOWNLOAD="macos" ;;
  linux) OS_DOWNLOAD="ubuntu" ;;
  *) echo "❌ Unsupported OS: $OS"; exit 1 ;;
esac

case "$PLATFORM" in
  aarch64) PLATFORM_DOWNLOAD="arm64" ;;
  x86_64) PLATFORM_DOWNLOAD="x64" ;;
  *) echo "❌ Unsupported platform: $PLATFORM"; exit 1 ;;
esac

# Construct the download URL using the mapped OS and PLATFORM
ZIP_NAME="llama-${VERSION}-bin-${OS_DOWNLOAD}-${PLATFORM_DOWNLOAD}.zip"
DOWNLOAD_URL="https://github.com/ggml-org/llama.cpp/releases/download/${VERSION}/${ZIP_NAME}"
TMP_DIR="$(mktemp -d)"

# Define the output directory based on the original OS and PLATFORM
OUTPUT_DIR="$DESTINATION/$OS/$PLATFORM"

# Create the output directory
mkdir -p "$OUTPUT_DIR"

echo "📥 Downloading llama.cpp version $VERSION for $OS/$PLATFORM..."
echo "🔗 $DOWNLOAD_URL"

# Download the file
curl -k -L -o "$TMP_DIR/$ZIP_NAME" "$DOWNLOAD_URL"

# Extract only the necessary files
echo "📦 Extracting libraries to $OUTPUT_DIR..."

unzip -q "$TMP_DIR/$ZIP_NAME" -d "$OUTPUT_DIR"

if [[ "$OS" == "macosx" ]]; then
  find "$OUTPUT_DIR" \( -type f -o -type l \) \( -name "*.dylib" -o -name "LICENSE" -o -name "LICENSE-*" \) | while read -r file; do
    mv "$file" "$OUTPUT_DIR/"
  done
  # Remove subdirectories if any remain
  find "$OUTPUT_DIR" -mindepth 1 -type d -exec rm -rf {} +
elif [[ "$OS" == "linux" ]]; then
  find "$OUTPUT_DIR" \( -type f -o -type l \) \( -name "*.so" -o -name "LICENSE" -o -name "LICENSE-*" \) | while read -r file; do
    mv "$file" "$OUTPUT_DIR/"
  done
  find "$OUTPUT_DIR" -mindepth 1 -type d -exec rm -rf {} +
else
  echo "❌ Unsupported OS: $OS"
  exit 1
fi

echo "✅ Downloaded and extracted libraries to $OUTPUT_DIR"
