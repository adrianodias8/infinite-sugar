#!/usr/bin/env bash
# Compile the WebAssembly LIF kernel (web/kernel/lif.c -> web/kernel/lif.wasm). Needs clang with
# the wasm32 target and wasm-ld (Ubuntu: clang + lld). The .wasm is committed, so this is only
# needed after editing lif.c.
set -euo pipefail
cd "$(dirname "$0")/../web/kernel"
clang --target=wasm32 -O3 -nostdlib -fno-builtin -Wl,--no-entry -Wl,--export-all -Wl,--import-memory -Wl,--allow-undefined -o lif.wasm lif.c
ls -la lif.wasm
