# wgpuweb

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

## Headless screenshot test

```sh
npm install     # one-time
npx playwright install chromium
npm test
```

`tests/screenshot.mjs` boots a Node http server for `web/dist/`, launches
headless Chromium with WebGPU enabled, and runs three checks:

1. **Pinned-baseline diff.** Loads with `?t=2.5&mx=0.5&my=0.5` — URL params
   the wasm reads to freeze `time` and pin the cursor — and mean-abs-diffs
   the locked frame against `tests/baseline.png`. Catches shader regressions
   and surface format drift.
2. **Mouse-API responsiveness.** With time still locked, screenshots before
   and after `page.mouse.move(...)` and asserts the image moved.
3. **Console-error gate.** Any `[error]` / `[pageerror]` / `[netfail]` line
   fails the test outright.

Run `UPDATE_BASELINE=1 npm test` after intentional rendering changes to
regenerate `tests/baseline.png`. Diagnostics for each run land in
`tests/output/`.

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
- **The web canvas must already exist in the DOM with explicit `width`/`height`
  attributes** before winit binds to it. `winit::Window::inner_size()` on web
  races browser layout and returns 0×0 right after `create_window`, which would
  silently produce a 1×1 surface that CSS scales up to look correctly sized but
  renders only one fragment-shader sample.
