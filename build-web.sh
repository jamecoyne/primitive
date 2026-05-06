#!/usr/bin/env bash
# Build the wasm32 target and run wasm-bindgen so it can be loaded as an ES module.
# Output lands in web/dist/ alongside the static index.html.
set -euo pipefail

cd "$(dirname "$0")"

rustup target add wasm32-unknown-unknown >/dev/null

# Resolve the wasm-bindgen version actually in use so the CLI matches.
if [ ! -f Cargo.lock ]; then
    cargo generate-lockfile
fi
WBG_VERSION=$(awk '
    /^\[\[package\]\]/ { in_pkg=1; name=""; ver=""; next }
    in_pkg && /^name = "wasm-bindgen"$/ { name="wasm-bindgen"; next }
    in_pkg && /^version = / && name=="wasm-bindgen" { gsub(/version = |"/,""); print; exit }
' Cargo.lock)

if [ -z "${WBG_VERSION:-}" ]; then
    echo "could not determine wasm-bindgen version from Cargo.lock" >&2
    exit 1
fi

INSTALLED=""
if command -v wasm-bindgen >/dev/null 2>&1; then
    INSTALLED=$(wasm-bindgen --version | awk '{print $2}')
fi

if [ "$INSTALLED" != "$WBG_VERSION" ]; then
    echo ">> installing wasm-bindgen-cli $WBG_VERSION (had: '${INSTALLED:-none}')"
    cargo install -f wasm-bindgen-cli --version "$WBG_VERSION"
fi

echo ">> cargo build --release --target wasm32-unknown-unknown --lib"
cargo build --release --target wasm32-unknown-unknown --lib

mkdir -p web/dist
echo ">> wasm-bindgen → web/dist/"
wasm-bindgen \
    --target web \
    --no-typescript \
    --out-dir web/dist \
    target/wasm32-unknown-unknown/release/wgpuweb.wasm

cp web/index.html web/dist/index.html

cat <<EOF

built. to serve:
  cd web/dist && python3 -m http.server 8000
then open http://localhost:8000
EOF
