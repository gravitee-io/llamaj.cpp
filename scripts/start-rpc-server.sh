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

# Downloads the prebuilt rpc-server binary from llama.cpp releases and runs it.
# Auto-detects OS (macOS / Linux) and architecture (arm64 / x86_64).
#
# On macOS, defaults to Metal-only (MTL0) to avoid BLAS backend crashes.
#
# The binary is cached in $HOME/.llama.cpp/rpc-server/ and reused on subsequent runs.
# Delete that directory to force a re-download.
#
# Usage:
#   ./scripts/start-rpc-server.sh                          # defaults: 0.0.0.0:50052
#   ./scripts/start-rpc-server.sh -p 50053                 # custom port
#   ./scripts/start-rpc-server.sh -H 127.0.0.1 -p 50052   # bind to localhost only
#   ./scripts/start-rpc-server.sh -d CPU                   # use CPU device only
#   ./scripts/start-rpc-server.sh -v b8000                 # custom llama.cpp version

set -euo pipefail

VERSION="b7943"
HOST="0.0.0.0"
PORT="50052"
DEVICE=""
INSTALL_DIR="$HOME/.llama.cpp/rpc-server"

print_usage() {
  echo "Usage: $0 [-v version] [-H host] [-p port] [-d device]"
  echo ""
  echo "Options:"
  echo "  -v <version>   llama.cpp release tag (default: ${VERSION})"
  echo "  -H <host>      Host to bind to (default: ${HOST})"
  echo "  -p <port>      Port to bind to (default: ${PORT})"
  echo "  -d <device>    Device to use (e.g., MTL0, CPU). Default: MTL0 on macOS, none on Linux"
  echo "  -h             Show this help"
}

while getopts ":v:H:p:d:h" opt; do
  case ${opt} in
    v) VERSION=$OPTARG ;;
    H) HOST=$OPTARG ;;
    p) PORT=$OPTARG ;;
    d) DEVICE=$OPTARG ;;
    h) print_usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; print_usage; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; print_usage; exit 1 ;;
  esac
done

# ---- Detect OS ----
case "$(uname -s)" in
  Darwin) OS_DOWNLOAD="macos" ;;
  Linux)  OS_DOWNLOAD="ubuntu" ;;
  *)      echo "Unsupported OS: $(uname -s)"; exit 1 ;;
esac

# ---- Detect architecture ----
case "$(uname -m)" in
  arm64|aarch64) ARCH_DOWNLOAD="arm64" ;;
  x86_64)        ARCH_DOWNLOAD="x64" ;;
  *)             echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

# ---- Default device: MTL0 on macOS to avoid BLAS backend crashes ----
if [ -z "$DEVICE" ] && [ "$OS_DOWNLOAD" = "macos" ]; then
  DEVICE="MTL0"
fi

RPC_BINARY="$INSTALL_DIR/rpc-server"

# ---- Download if not already cached ----
if [ ! -x "$RPC_BINARY" ]; then
  ARCHIVE_NAME="llama-${VERSION}-bin-${OS_DOWNLOAD}-${ARCH_DOWNLOAD}.tar.gz"
  DOWNLOAD_URL="https://github.com/ggml-org/llama.cpp/releases/download/${VERSION}/${ARCHIVE_NAME}"

  echo "Downloading llama.cpp ${VERSION} rpc-server (${OS_DOWNLOAD}/${ARCH_DOWNLOAD})..."
  echo "  ${DOWNLOAD_URL}"

  TMP_DIR="$(mktemp -d)"
  trap "rm -rf $TMP_DIR" EXIT

  curl -fSL -o "$TMP_DIR/$ARCHIVE_NAME" "$DOWNLOAD_URL"

  mkdir -p "$INSTALL_DIR"
  tar -xzf "$TMP_DIR/$ARCHIVE_NAME" -C "$TMP_DIR"

  # Find and copy the rpc-server binary
  RPC_BIN=$(find "$TMP_DIR" -name "rpc-server" -type f | head -1)
  if [ -z "$RPC_BIN" ]; then
    echo "Error: rpc-server binary not found in release archive."
    exit 1
  fi
  cp "$RPC_BIN" "$INSTALL_DIR/"

  # Copy shared libraries alongside the binary
  if [ "$OS_DOWNLOAD" = "macos" ]; then
    find "$TMP_DIR" -name "*.dylib" -exec cp {} "$INSTALL_DIR/" \;
  else
    find "$TMP_DIR" -name "*.so*" -exec cp {} "$INSTALL_DIR/" \;
  fi

  chmod +x "$RPC_BINARY"
  echo "Installed rpc-server to $INSTALL_DIR"
fi

# ---- Build command ----
CMD="./rpc-server -H $HOST -p $PORT"
if [ -n "$DEVICE" ]; then
  CMD="$CMD -d $DEVICE"
fi

# ---- Run from the install dir so the binary finds its shared libs ----
echo "Starting rpc-server on ${HOST}:${PORT} (device: ${DEVICE:-all})..."
cd "$INSTALL_DIR"
exec $CMD
