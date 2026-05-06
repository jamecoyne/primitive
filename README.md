# wgpuweb

[![Test](https://github.com/jamecoyne/primitive/actions/workflows/test.yml/badge.svg)](https://github.com/jamecoyne/primitive/actions/workflows/test.yml)
[![Deploy to GitHub Pages](https://github.com/jamecoyne/primitive/actions/workflows/pages.yml/badge.svg)](https://github.com/jamecoyne/primitive/actions/workflows/pages.yml)

Proof-of-concept Rust + [wgpu](https://wgpu.rs) app that renders the Mandelbrot
set with a GLSL fragment shader and runs from the same source on **native
macOS** and **the browser** (compiled to WebAssembly via `wasm-bindgen`). The
mandelbrot's center tracks the cursor.

![mandelbrot screenshot](tests/baseline.png)

## Architecture

- Single crate (`src/lib.rs` is shared; `src/main.rs` is the native entry,
  `web_main()` annotated with `#[wasm_bindgen(start)]` is the web entry).
- [winit 0.30](https://docs.rs/winit) `ApplicationHandler` for the event loop on
  both targets.
- wgpu 24, with the `glsl` feature so naga can ingest `#version 450` GLSL and
  cross-compile it to MSL on Mac and SPIR-V → WebGPU in the browser. The
  `webgl` feature is enabled on `wasm32` so browsers without WebGPU still get a
  WebGL2 fallback.
- Async device init is bridged across platforms with an `EventLoopProxy<UserEvent>`:
  the wasm path `spawn_local`s the future and posts `StateReady(state)` back to
  the event loop; the native path uses `pollster::block_on`.
- Fullscreen-triangle vertex shader + Mandelbrot fragment shader. Uniforms:
  `resolution`, `mouse` (pixels), `time`. Mouse → NDC conversion handles the
  winit y-down / NDC y-up flip.

## Quick start

### Native (macOS, Metal)

```sh
cargo run --release
```

### Web (WebGPU, with WebGL2 fallback)

```sh
./build-web.sh
cd web/dist && python3 -m http.server 8000
# open http://localhost:8000
```

`build-web.sh` adds the `wasm32-unknown-unknown` rustup target and installs a
`wasm-bindgen-cli` whose version matches the `wasm-bindgen` crate in
`Cargo.lock` (the two must match exactly).

## Test harness

```sh
npm install                        # one-time
npx playwright install chromium    # one-time
npm test                           # native + web + cross-platform diff
npm run test:native                # cargo test --test headless
npm run test:web                   # rebuild wasm + Playwright screenshot
npm run test:cross                 # strict pixel-perfect cross-platform diff
```

Native and web run the same three checks against their own platform's
rendering pipeline:

1. **Pinned-baseline diff.** A locked input (`time = 2.5`, `mouse_uv = (0.5, 0.5)`)
   should produce a deterministic frame. Mean abs RGB diff must be ≤ 8/255
   vs `tests/baseline.png` (web, browser screenshot) or
   `tests/baseline-native.png` (native, `render_offscreen` readback). Catches
   shader regressions and surface-format drift.
2. **Mouse-API responsiveness.** Image must change by ≥ 12/255 when the
   cursor moves — proves the mouse uniform is plumbed through.
3. **Error gate.** Web fails on any `[error]` / `[pageerror]` / `[netfail]`;
   native fails on any panic via `cargo test`. Mostly catches naga
   translation errors and validation panics the moment they appear.

Each platform has its own baseline. They render byte-identical pixels at
the same locked input — verified by `npm run test:cross`, which
strict-compares the native `render_offscreen` readback against the web
canvas screenshot and fails on any single pixel diff. Achieving zero diff
required two non-obvious fixes: configuring the surface with an sRGB view
format on web (Chrome only exposes non-sRGB storage formats), and
suppressing the browser-injected canvas focus outline before screenshot.
See `CLAUDE.md` for details.

Run `UPDATE_BASELINE=1 npm test` after intentional rendering changes and
commit both baselines. Diagnostics for the web runs land in `tests/output/`.

## Project layout

```
Cargo.toml
src/
  lib.rs             # shared app + State + ApplicationHandler
  main.rs            # native entry
shaders/
  mandelbrot.vert    # fullscreen triangle
  mandelbrot.frag    # iterated z² + c with smooth coloring
web/
  index.html         # canvas + module loader (carried into web/dist by build-web.sh)
build-web.sh         # wasm32 → web/dist/ with version-matched wasm-bindgen-cli
tests/
  screenshot.mjs     # Playwright + pngjs harness
CLAUDE.md            # operating notes for Claude Code (build commands, gotchas)
```

## Notes / gotchas

- **wgpu must be ≥24.** wgpu 22 sends the now-removed WebGPU spec field
  `maxInterStageShaderComponents`, which modern Chromium rejects at
  `requestDevice`.
- **The web canvas's drawing-buffer dimensions must be set in JS before
  the wasm reads them.** `index.html`'s inline script does
  `c.width = window.innerWidth; c.height = window.innerHeight` before
  `await init()`. Without this `winit::Window::inner_size()` races browser
  layout and returns 0×0, which previously produced a 1×1 surface that CSS
  scaled up to look right but rendered only one fragment-shader sample.
