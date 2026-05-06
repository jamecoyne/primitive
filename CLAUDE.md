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

## Test harness

```
npm test                          # native + web + cross-platform
npm run test:native               # cargo test --test headless
npm run test:web                  # ./build-web.sh + Playwright screenshot harness
npm run test:cross                # strict pixel-perfect cross-platform diff
UPDATE_BASELINE=1 npm test        # regenerate native + web baselines
```

The native and web halves run the same three platform-internal checks
(pinned baseline, mouse responsiveness, error gate). The cross-platform
test asserts strict pixel equality between them. After the surface-format
fix in `lib.rs` and the canvas-chrome suppression in `cross_platform.mjs`,
the diff is 0/691,200 — both platforms produce byte-identical frames at
the same locked input.

### Native (`tests/headless.rs`, `cargo test`)

`wgpuweb::render_offscreen(RenderConfig { width, height, time, mouse_uv })`
is a public lib function that builds a wgpu instance with no surface,
renders one frame to an `Rgba8UnormSrgb` texture, copies to a staging buffer,
and returns tightly-packed RGBA8. `tests/headless.rs` uses it for:

1. `pinned_baseline` — diff vs `tests/baseline-native.png` ≤ 8/255.
2. `mouse_changes_image` — render at `mouse_uv [0.5, 0.5]` and `[0.1, 0.1]`,
   assert diff ≥ 12/255.
3. Panic gate is implicit — `cargo test` fails on any panic during render
   setup, equivalent to the browser's console-error gate.

`UPDATE_BASELINE=1 cargo test --test headless pinned_baseline` regenerates
the file. Commit it.

### Cross-platform (`tests/cross_platform.mjs`)

Strict pixel-perfect equality between the native and web frames at the same
locked input. The native side renders via `cargo run --bin render_frame` and
the web side captures the canvas via Playwright. Any single pixel differing
in any channel fails the test.

Two non-obvious things are needed to actually hit zero diff:

1. **Surface format**: Chrome's WebGPU canvas only exposes non-sRGB storage
   formats (`Bgra8Unorm`, `Rgba8Unorm`, `Rgba16Float`). Without intervention,
   `surface.get_capabilities().formats[0]` is non-sRGB and the linear shader
   output gets written raw, while native picks an sRGB format and applies
   the encoding — diff was 91% before fixing. `lib.rs` now configures the
   non-sRGB storage with a `view_format` of the sRGB sibling and creates
   the render-pass view with that format, so the encoding step happens on
   both platforms regardless of what the canvas exposes.
2. **Browser-injected canvas chrome**: Chromium paints a 1-pixel focus
   outline (`rgb(0, 95, 204)` macOS system blue) on the canvas element,
   which writes into the captured perimeter. The test's `addInitScript`
   strips `box-shadow`, `outline`, and `border` on `canvas#wgpu` before
   first paint. Don't touch this without re-running the test.

### Web (`tests/screenshot.mjs`, Playwright)

`tests/screenshot.mjs` runs three checks. `npm run test:web` rebuilds the
wasm first; if invoking the script directly, run `./build-web.sh` first.

1. **Pinned-baseline diff.** Loads `?t=2.5&mx=0.5&my=0.5` — those URL params
   are read by `parse_test_lock_from_url()` in `lib.rs` and freeze the time
   uniform + pin the cursor. The locked frame is mean-abs-diffed against
   `tests/baseline.png` (committed); fail threshold `8/255`. WebGPU is
   deterministic on the same machine — actual diff is `0.00/255`. Bump the
   threshold if you observe drift across machines.
2. **Mouse-API responsiveness.** Loads `?t=2.5` (time locked, mouse free),
   takes a screenshot, moves the cursor, takes another. Mean abs diff must be
   `≥ 12/255`. Time lock means the diff reflects only the cursor move, not
   color/zoom drift.
3. **Console-error gate.** Any `[error]`, `[pageerror]`, or `[netfail]` line
   collected during the run fails the test. This catches things like the
   wgpu 22 `maxInterStageShaderComponents` panic the moment they appear,
   without needing to reason about pixel statistics.

Sanity bounds (`avgLuma ≥ 8`, combined RGB range `≥ 30`) still apply to the
locked frame as a black/uniform-color guard.

Outputs (always written to `tests/output/`, gitignored):
- `locked.png` — pinned-baseline frame.
- `before-mouse.png` / `after-mouse.png` — mouse-API frames.
- `console.log` — full browser console + pageerror + netfail capture.

When you intentionally change rendering (shader edit, surface format, etc.),
run `UPDATE_BASELINE=1 npm test`, eyeball the new `tests/baseline.png`, and
commit it.

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
- **Always render through the sRGB view of the surface texture.**
  `State::new` configures the surface with `view_formats: [sRGB sibling]`
  and `State::render` builds the color-attachment view with
  `format: Some(view_format)`. Removing this would silently break
  cross-platform pixel parity on the web — Chrome's WebGPU canvas exposes
  only non-sRGB storage formats, so without this fix the linear shader
  output is written raw on web while native picks an sRGB format and
  applies the encoding. Pre-fix diff: 91% of pixels differ; post-fix: 0.
