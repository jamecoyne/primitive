# wgpuweb

Rust + wgpu Mandelbrot rendered via GLSL, building for native macOS and the
browser (WASM). Single crate, both targets share `src/lib.rs`; `src/main.rs` is
the native entry point and `web_main()` (annotated `#[wasm_bindgen(start)]`) is
the web entry point.

## Build & run

| Target | Command | Notes |
|---|---|---|
| Native (Mac) | `cargo run --release` | Metal backend, opens a winit window |
| Web | `./build-web.sh` | Outputs to `web/dist/` |
| Serve web build | `cd web/dist && python3 -m http.server 8000` | Then open http://localhost:8000 |

`build-web.sh` reads the `wasm-bindgen` version from `Cargo.lock` and
auto-installs a matching `wasm-bindgen-cli` if needed. Versions must match
exactly or the wasm fails to load.

## Screenshot test (web)

```
npm test
```

Runs `tests/screenshot.mjs`:

1. Serves `web/dist/` in-process on a random port.
2. Launches headless Chromium via Playwright with WebGPU enabled
   (`--enable-unsafe-webgpu --use-angle=metal`).
3. Waits ~2.5 s for wasm init + first frames, then screenshots **only the
   canvas region** (cropped via `getBoundingClientRect`) — the page chrome
   would otherwise mask a fully-black canvas and hide bugs.
4. Parses the PNG with `pngjs` and asserts:
   - `avgLuma ≥ 8` (catches all-black canvas)
   - combined RGB range `≥ 30` (catches uniform-color canvas)
5. Moves the cursor to a different canvas location, screenshots again, and
   asserts mean abs RGB diff `≥ 12/255`. The mandelbrot's center tracks the
   cursor (see GLSL fragment shader), so a working mouse API moves the image
   substantially; the threshold sits comfortably above time-based color/zoom
   drift between two ~400ms screenshots.

Diagnostics on every run:
- `tests/output/canvas.png` — baseline canvas screenshot (cursor at default)
- `tests/output/canvas-mouse.png` — after cursor moved
- `tests/output/screenshot.png` — full-page screenshot
- `tests/output/console.log` — browser console + page errors + failed requests

Always rebuild with `./build-web.sh` before `npm test`; the test reads the
last-built `web/dist/` and will fail with exit 2 if the build is missing.

## Deployment

`main` auto-deploys to GitHub Pages via `.github/workflows/pages.yml`.
Live URL: https://jamecoyne.github.io/primitive/ (publicly viewable even
though the source repo is private — standard GitHub Pages behavior).

The workflow extracts the wasm-bindgen version from `Cargo.lock` and
pre-installs the matching CLI via `taiki-e/install-action` so build-web.sh
skips the slow `cargo install` step. Don't break the awk version-extraction
without updating both the workflow and `build-web.sh`.

## Gotchas — do not regress

- **wgpu version is pinned to 24.x.** wgpu 22.x sends the removed WebGPU
  spec field `maxInterStageShaderComponents`, which modern Chromium rejects
  at `requestDevice` with `OperationError`. Don't downgrade. wgpu ≥25 has a
  larger API delta (many breaking changes); upgrading is fine but expect
  several call-site updates.
- **The web canvas must already exist in the DOM at a known size before
  winit binds to it.** `index.html` has `<canvas id="wgpu" width="960"
  height="720">` and `lib.rs` looks it up via `get_element_by_id` and
  `WindowAttributesExtWebSys::with_canvas`. Don't replace this with
  `with_append(true)` — `winit::Window::inner_size()` on web races browser
  layout and returns 0×0 immediately after window creation, which produced a
  1×1 surface that was CSS-scaled up to look correctly sized but rendered a
  single fragment-shader sample.
- The initial size is threaded into `State::new(window, initial_size)`
  directly rather than being read from `window.inner_size()`, for the same
  reason.
