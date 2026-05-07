//! Native test harness — mirrors the three checks the browser test does in
//! `tests/screenshot.mjs`, but runs against the same wgpu pipeline directly
//! through `wgpuweb::render_offscreen` instead of going through winit and a
//! window surface.
//!
//! Checks:
//!   1. `pinned_baseline`        — locked time + mouse → mean abs diff vs
//!                                  tests/baseline-native.png ≤ MAX_BASELINE_DIFF.
//!   2. `mouse_changes_image`    — moving the mouse uv changes the rendered
//!                                  image by ≥ MIN_MOUSE_DIFF.
//!   3. (implicit) panic gate    — `cargo test` fails if any panic occurs
//!                                  during render setup; equivalent to the
//!                                  console-error gate on the browser side.
//!
//! Use `UPDATE_BASELINE=1 cargo test --test headless pinned_baseline` to
//! regenerate the committed baseline after intentional rendering changes.

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use wgpuweb::render_graph::{RenderGraph, DEFAULT_GRAPH_TOML};
use wgpuweb::{render_offscreen, RenderConfig};

const W: u32 = 960;
const H: u32 = 720;
const BASELINE_PATH: &str = "tests/baseline-native.png";
const MAX_BASELINE_DIFF: f64 = 8.0;
const MIN_MOUSE_DIFF: f64 = 12.0;

fn render(time: f32, mouse_uv: [f32; 2]) -> Vec<u8> {
    pollster::block_on(render_offscreen(RenderConfig {
        width: W,
        height: H,
        time,
        mouse_uv,
    }))
}

fn write_png<P: AsRef<Path>>(path: P, pixels: &[u8]) {
    let file = File::create(&path).expect("create png");
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, W, H);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    writer.write_image_data(pixels).expect("png write");
}

fn read_png<P: AsRef<Path>>(path: P) -> Vec<u8> {
    let file = File::open(&path).expect("open baseline png");
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().expect("png info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(
        info.color_type,
        png::ColorType::Rgba,
        "baseline png must be RGBA8"
    );
    buf
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "frame size mismatch");
    let mut sum: u64 = 0;
    let mut count: u64 = 0;
    let mut i = 0;
    while i < a.len() {
        sum += a[i].abs_diff(b[i]) as u64
            + a[i + 1].abs_diff(b[i + 1]) as u64
            + a[i + 2].abs_diff(b[i + 2]) as u64;
        count += 3;
        i += 4;
    }
    sum as f64 / count as f64
}

#[test]
fn pinned_baseline() {
    let pixels = render(2.5, [0.5, 0.5]);

    if env::var("UPDATE_BASELINE").as_deref() == Ok("1") {
        write_png(BASELINE_PATH, &pixels);
        eprintln!("UPDATE_BASELINE=1: wrote {BASELINE_PATH}");
        return;
    }

    let baseline = match File::open(BASELINE_PATH) {
        Ok(_) => read_png(BASELINE_PATH),
        Err(_) => panic!(
            "{BASELINE_PATH} missing — run `UPDATE_BASELINE=1 cargo test --test headless` to create it"
        ),
    };

    let diff = mean_abs_diff(&pixels, &baseline);
    eprintln!("native baseline diff: {diff:.2}/255");
    assert!(
        diff <= MAX_BASELINE_DIFF,
        "native baseline diff {diff:.2} > {MAX_BASELINE_DIFF}/255 — \
         shader change? regenerate via `UPDATE_BASELINE=1 cargo test --test headless`"
    );
}

#[test]
fn mouse_changes_image() {
    let centered = render(2.5, [0.5, 0.5]);
    let off_center = render(2.5, [0.1, 0.1]);
    let diff = mean_abs_diff(&centered, &off_center);
    eprintln!("native mouse-induced diff: {diff:.2}/255");
    assert!(
        diff >= MIN_MOUSE_DIFF,
        "moving mouse_uv from center to (0.1, 0.1) only changed image by {diff:.2}/255 \
         (need ≥ {MIN_MOUSE_DIFF}); mouse uniform might not be applied"
    );
}

/// Build the default graph and assert that every node with `viewer.enabled =
/// true` in `config/graph.toml` shows up in `viewer_textures()` at the
/// configured resolution. Surfaces a regression if either the schema parse
/// or the per-node texture allocation breaks.
#[test]
fn viewer_textures_match_config() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no adapter");
        let (device, _queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("viewer-test-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("request_device");

        let mut graph = RenderGraph::from_toml(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            DEFAULT_GRAPH_TOML,
        )
        .expect("graph builds from default TOML");
        graph.resize(&device, 1024, 800);

        let viewers = graph.viewer_textures();
        // Default config has the mandelbrot viewer enabled at 256×192.
        let mandel = viewers
            .iter()
            .find(|v| v.node_id == "mandelbrot")
            .expect("mandelbrot viewer slot missing");
        assert_eq!(
            (mandel.width, mandel.height),
            (256, 192),
            "mandelbrot viewer resolution should match `viewer.resolution` in graph.toml"
        );

        // Anything else we declared `viewer.enabled = false` (or omitted)
        // must NOT appear here.
        for v in &viewers {
            assert_ne!(
                v.node_id, "invert",
                "invert has no `viewer.enabled = true` so it shouldn't appear in viewer_textures"
            );
        }
    });
}
