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
npm test                          # native + web
npm run test:native               # cargo test (~0.05s after compile)
npm run test:web                  # ./build-web.sh + screenshot harness
UPDATE_BASELINE=1 npm test        # regenerate both baselines
```

The native and web halves use the same three checks (pinned baseline,
mouse responsiveness, error gate) but each maintains its **own** baseline
image because the rendering paths differ — native goes naga→MSL via Metal,
web goes naga→WGSL via the browser, and they produce subtly different
floats. Don't try to share `tests/baseline.png` and `tests/baseline-native.png`.

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
