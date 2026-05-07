use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod hud;
pub mod render_graph;
use hud::Hud;
use render_graph::{NodeContext, RenderGraph, ViewerRegion, DEFAULT_GRAPH_TOML};

/// Test-mode overrides parsed from URL params on web. None on native.
/// When `time` is Some, the time uniform is frozen at that value. When
/// `mouse_uv` is Some, the cursor is pinned and CursorMoved events are
/// ignored — used by the screenshot harness to render deterministic frames.
#[derive(Clone, Copy, Default)]
struct TestLock {
    time: Option<f32>,
    mouse_uv: Option<[f32; 2]>,
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    graph: RenderGraph,
    // Format used for the render-pass color attachment. May differ from
    // config.format: on web the WebGPU canvas only accepts non-sRGB
    // storage formats, so we render through an sRGB view of the canvas
    // texture to keep the linear→sRGB encoding consistent with native.
    view_format: wgpu::TextureFormat,
    start: web_time::Instant,
    mouse: [f32; 2],
    lock: TestLock,
    // HUD overlay — adapter info, allocated bytes, pass count, rolling
    // FPS, and rolling CPU time per render() call. Hidden when `lock`
    // has any field set so the headless screenshot harness can still
    // produce byte-identical frames.
    hud: Hud,
    adapter_info: wgpu::AdapterInfo,
    last_frame: Option<web_time::Instant>,
    /// Ring of recent frame intervals in seconds (most recent at the back).
    /// Capped at FRAME_WINDOW; FPS is N / sum(window).
    frame_times: std::collections::VecDeque<f32>,
    /// Ring of recent render() durations in seconds. Same window as above.
    cpu_times: std::collections::VecDeque<f32>,
    /// Pass count from the most recent `graph.render` call. Updated each
    /// frame; not smoothed because it's deterministic from the config.
    last_pass_count: u32,
    /// Latest cursor position from `WindowEvent::CursorMoved` (physical
    /// pixels). Tracked even while dragging so we always know where the
    /// pointer is on `MouseInput`.
    cursor_pos: [f32; 2],
    /// Active viewer-thumbnail drag, if any. While `Some`, the cursor's
    /// CursorMoved updates the dragged viewer's position instead of the
    /// mandelbrot uniform. Cleared on left-button release.
    dragging: Option<DragState>,
}

#[derive(Clone, Copy)]
struct DragState {
    /// Index into `RenderGraph::viewer_textures()` (== node index).
    viewer_index: usize,
    /// Cursor → viewer-top-left offset captured at drag start, in
    /// physical pixels. Subtracted from cursor each frame to keep the
    /// drag-grab point under the pointer.
    grab_offset: [f32; 2],
}

const FRAME_WINDOW: usize = 60;

/// Returns the sRGB-encoding sibling of a format if one exists, else the
/// format itself. Lets us write linear shader output through an sRGB view
/// even when the underlying canvas storage is non-sRGB.
fn srgb_view_of(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    use wgpu::TextureFormat::*;
    match format {
        Bgra8Unorm => Bgra8UnormSrgb,
        Rgba8Unorm => Rgba8UnormSrgb,
        other => other,
    }
}

impl State {
    async fn new(
        window: Arc<Window>,
        initial_size: PhysicalSize<u32>,
        lock: TestLock,
    ) -> Self {
        let size = PhysicalSize::new(initial_size.width.max(1), initial_size.height.max(1));
        log::info!("State::new size = {:?}", size);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("create_surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no compatible adapter");

        let adapter_info = adapter.get_info();
        log::info!("adapter: {:?}", adapter_info);
        log::info!(
            "rendering on {} via {}",
            hud::device_type_label(adapter_info.device_type),
            hud::backend_label(adapter_info.backend),
        );
        if adapter_info.device_type == wgpu::DeviceType::Cpu {
            log::warn!("wgpu picked a CPU adapter — work is NOT running on the GPU");
        }

        let required_limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits::downlevel_webgl2_defaults()
        } else {
            wgpu::Limits::default()
        };

        // Opt in to TIMESTAMP_QUERY only when the adapter advertises it.
        // Hardware-accelerated Metal/Vulkan/D3D12 always do; Chrome's
        // WebGPU exposes it behind a runtime flag; older WebGL2 paths
        // never will. Without it, the HUD's GPU-cook column stays "—".
        let mut required_features = wgpu::Features::empty();
        let supports_timestamps = adapter
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY);
        if supports_timestamps {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        log::info!("TIMESTAMP_QUERY: {}", supports_timestamps);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features,
                    required_limits,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("request_device");

        let caps = surface.get_capabilities(&adapter);
        // Prefer an sRGB storage format; if none (WebGPU canvases on Chrome
        // expose only Bgra8Unorm/Rgba8Unorm), keep the non-sRGB storage and
        // we'll render through an sRGB view below to keep linear→sRGB
        // encoding consistent across platforms.
        let storage_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let view_format = srgb_view_of(storage_format);

        let view_formats = if storage_format != view_format {
            vec![view_format]
        } else {
            vec![]
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: storage_format,
            width: size.width,
            height: size.height,
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats,
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut graph = RenderGraph::from_toml(&device, view_format, DEFAULT_GRAPH_TOML)
            .unwrap_or_else(|e| panic!("config/graph.toml: {e}"));
        // Allocate intermediates + bind groups for the initial canvas size.
        // resize() must run before the first render.
        graph.resize(&device, size.width, size.height);
        // Enable per-pass GPU timestamps if the device supports them.
        // No-ops on adapters that didn't advertise TIMESTAMP_QUERY.
        graph.enable_perf_monitor(&device, &queue);

        let hud = Hud::new(&device, view_format);

        let default_mouse = [size.width as f32 * 0.5, size.height as f32 * 0.5];
        let mouse = match lock.mouse_uv {
            Some([u, v]) => [u * size.width as f32, v * size.height as f32],
            None => default_mouse,
        };
        Self {
            window,
            surface,
            device,
            queue,
            config,
            graph,
            view_format,
            start: web_time::Instant::now(),
            mouse,
            lock,
            hud,
            adapter_info,
            last_frame: None,
            frame_times: std::collections::VecDeque::with_capacity(FRAME_WINDOW),
            cpu_times: std::collections::VecDeque::with_capacity(FRAME_WINDOW),
            last_pass_count: 0,
            cursor_pos: default_mouse,
            dragging: None,
        }
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        // Reallocate the graph's intermediate textures and rebuild bind
        // groups so downstream nodes sample from the new sizes.
        self.graph.resize(&self.device, size.width, size.height);
        // When the cursor is pinned via test lock, keep it pinned in uv space
        // even as the canvas resizes — otherwise locked-frame screenshots
        // would drift after the first ResizeObserver event on web.
        if let Some([u, v]) = self.lock.mouse_uv {
            self.mouse = [u * size.width as f32, v * size.height as f32];
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Frame timing — measure before doing any GPU work so the
        // wall-clock dt is roughly "frame to frame" from the renderer's
        // POV. Skipped on the very first frame (no prior timestamp).
        let now = web_time::Instant::now();
        if let Some(prev) = self.last_frame {
            let dt = now.duration_since(prev).as_secs_f32();
            // Guard against bad timestamps (shouldn't happen but cheap).
            if dt > 0.0 && dt < 1.0 {
                if self.frame_times.len() == FRAME_WINDOW {
                    self.frame_times.pop_front();
                }
                self.frame_times.push_back(dt);
            }
        }
        self.last_frame = Some(now);
        let render_start = now;

        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });

        let time = self
            .lock
            .time
            .unwrap_or_else(|| self.start.elapsed().as_secs_f32());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });
        let mut ctx = NodeContext {
            encoder: &mut encoder,
            queue: &self.queue,
            output: &view,
            width: self.config.width,
            height: self.config.height,
            time,
            mouse: self.mouse,
        };
        let stats = self.graph.render(&mut ctx);
        self.last_pass_count = stats.passes;

        // HUD overlay — render on top of the graph output. Hidden when a
        // test lock is set so the screenshot harness gets byte-identical
        // frames; otherwise frame-time and adapter-name jitter would
        // break cross-platform parity.
        if self.lock.time.is_none() && self.lock.mouse_uv.is_none() {
            let fps = if self.frame_times.is_empty() {
                0.0
            } else {
                let sum: f32 = self.frame_times.iter().sum();
                self.frame_times.len() as f32 / sum
            };
            let cpu_ms = if self.cpu_times.is_empty() {
                0.0
            } else {
                let sum: f32 = self.cpu_times.iter().sum();
                sum / self.cpu_times.len() as f32 * 1000.0
            };
            // The HUD pass adds one more render pass to whatever the
            // graph reported, so include it in the displayed total.
            let total_passes = self.last_pass_count + 1;
            let summary = self.graph.summary();
            let text = build_hud_text(
                &summary,
                &self.adapter_info,
                cpu_ms,
                fps,
                total_passes,
            );
            self.hud.set_text(&self.queue, &text);

            const HUD_MARGIN: u32 = 16;
            let hud_w = self.hud.width;
            let hud_h = self.hud.height;
            let canvas_w = self.config.width;
            let canvas_h = self.config.height;
            // Bottom-right anchor; skip if the HUD wouldn't fit.
            if hud_w + HUD_MARGIN <= canvas_w && hud_h + HUD_MARGIN <= canvas_h {
                let viewport = [
                    (canvas_w - hud_w - HUD_MARGIN) as f32,
                    (canvas_h - hud_h - HUD_MARGIN) as f32,
                    hud_w as f32,
                    hud_h as f32,
                ];
                self.hud.record(&mut encoder, &view, viewport, None);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();

        // Drive perf-monitor async readbacks. No-op when timestamps
        // aren't enabled or are still in flight.
        self.graph.poll_perf(&self.device);

        // Capture CPU work duration after submit. Smoothed across the same
        // 60-frame window as fps; fed back into the HUD on the *next* frame
        // (so the displayed value is from the previous render's work).
        let cpu_dt = render_start.elapsed().as_secs_f32();
        if cpu_dt > 0.0 && cpu_dt < 1.0 {
            if self.cpu_times.len() == FRAME_WINDOW {
                self.cpu_times.pop_front();
            }
            self.cpu_times.push_back(cpu_dt);
        }
        Ok(())
    }
}

#[allow(dead_code)] // StateReady is only constructed on wasm32
enum UserEvent {
    StateReady(State),
}

impl std::fmt::Debug for UserEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UserEvent::StateReady(_) => f.write_str("StateReady"),
        }
    }
}

struct App {
    #[allow(dead_code)] // only used on wasm32 to deliver async-built State
    proxy: Option<EventLoopProxy<UserEvent>>,
    state: Option<State>,
    lock: TestLock,
}

#[cfg(target_arch = "wasm32")]
fn parse_test_lock_from_url() -> TestLock {
    let Some(window) = web_sys::window() else {
        return TestLock::default();
    };
    let Ok(search) = window.location().search() else {
        return TestLock::default();
    };
    let qs = search.trim_start_matches('?');
    let (mut t, mut mx, mut my) = (None, None, None);
    for kv in qs.split('&') {
        let mut split = kv.splitn(2, '=');
        let k = split.next().unwrap_or("");
        let v = split.next().unwrap_or("");
        match k {
            "t" => t = v.parse::<f32>().ok(),
            "mx" => mx = v.parse::<f32>().ok(),
            "my" => my = v.parse::<f32>().ok(),
            _ => {}
        }
    }
    TestLock {
        time: t,
        mouse_uv: match (mx, my) {
            (Some(x), Some(y)) => Some([x, y]),
            _ => None,
        },
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("Primative")
            .with_inner_size(PhysicalSize::new(960u32, 720));

        #[cfg(target_arch = "wasm32")]
        let (attrs, initial_size) = {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("wgpu"))
                .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                .expect("canvas#wgpu not found in DOM");
            let size = PhysicalSize::new(canvas.width(), canvas.height());
            log::info!("canvas dims from DOM: {}x{}", canvas.width(), canvas.height());
            let attrs = attrs.with_inner_size(size).with_canvas(Some(canvas));
            (attrs, size)
        };

        #[cfg(not(target_arch = "wasm32"))]
        let initial_size = PhysicalSize::new(960u32, 720);

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        let lock = self.lock;

        #[cfg(target_arch = "wasm32")]
        {
            let proxy = self.proxy.take().expect("proxy taken");
            let window_clone = window.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let state = State::new(window_clone, initial_size, lock).await;
                let _ = proxy.send_event(UserEvent::StateReady(state));
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = pollster::block_on(State::new(window, initial_size, lock));
            self.state = Some(state);
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::StateReady(state) => {
                state.window.request_redraw();
                self.state = Some(state);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size);
                state.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cursor = [position.x as f32, position.y as f32];
                state.cursor_pos = cursor;
                if let Some(drag) = state.dragging {
                    // Convert cursor → desired thumbnail top-left, clamp
                    // to canvas, push into the graph. The thumbnail's
                    // size doesn't change so we read it from
                    // `viewer_rect`.
                    if let Some([_, _, w, h]) = state.graph.viewer_rect(drag.viewer_index) {
                        let nx = (cursor[0] - drag.grab_offset[0]).max(0.0) as u32;
                        let ny = (cursor[1] - drag.grab_offset[1]).max(0.0) as u32;
                        let max_x = state.config.width.saturating_sub(w);
                        let max_y = state.config.height.saturating_sub(h);
                        state.graph.set_viewer_position(
                            drag.viewer_index,
                            [nx.min(max_x), ny.min(max_y)],
                        );
                    }
                } else if state.lock.mouse_uv.is_none() {
                    state.mouse = cursor;
                }
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: button_state,
                ..
            } => {
                // Tests use `lock.mouse_uv` to pin the cursor; if that's
                // set, the headless path mustn't react to mouse buttons
                // either or the rendered output stops being deterministic.
                if state.lock.mouse_uv.is_some() {
                    return;
                }
                match button_state {
                    ElementState::Pressed => {
                        let cu = [state.cursor_pos[0] as u32, state.cursor_pos[1] as u32];
                        match state.graph.hit_test_viewer_region(cu) {
                            Some((idx, ViewerRegion::Eye)) => {
                                // Toggle the per-viewer preview. Don't
                                // start a drag; the user is interacting
                                // with the icon, not the thumbnail body.
                                let _ = state.graph.toggle_viewer_preview(idx);
                            }
                            Some((idx, ViewerRegion::Body)) => {
                                if let Some([rx, ry, _, _]) = state.graph.viewer_rect(idx) {
                                    state.dragging = Some(DragState {
                                        viewer_index: idx,
                                        grab_offset: [
                                            state.cursor_pos[0] - rx as f32,
                                            state.cursor_pos[1] - ry as f32,
                                        ],
                                    });
                                }
                            }
                            None => {}
                        }
                    }
                    ElementState::Released => {
                        state.dragging = None;
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                match state.render() {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => log::warn!("render error: {e:?}"),
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Build the multi-line HUD text from a graph summary + per-frame
/// telemetry. The right column shows GPU cook ms when the perf monitor
/// is producing samples, otherwise CPU dispatch ms (the encoder-build
/// time, near-zero) — the column header reflects which.
///
/// ```text
/// RENDER GRAPH         MB    GPU ms
/// mandelbrot v         3.13  0.42
/// +- invert *  v       3.13  0.05
/// total: 6.26 MiB / 8 passes
/// GPU:     <adapter name>
/// Type:    <integrated/discrete/cpu>
/// Backend: <backend>  CPU: 0.31 ms
/// FPS:     60.0
/// ```
///
/// `*` = present node, `v` = `viewer.enabled = true`.
fn build_hud_text(
    summary: &render_graph::GraphSummary,
    adapter_info: &wgpu::AdapterInfo,
    cpu_ms: f32,
    fps: f32,
    passes: u32,
) -> String {
    // Use GPU cook time when any node has a sample; otherwise fall back
    // to dispatch time so the column never reads "—".
    let has_gpu = summary.nodes.iter().any(|n| n.gpu_ms.is_some());
    let header_unit = if has_gpu { "GPU ms" } else { "REC ms" };

    let mut s = String::new();
    s.push_str(&format!("RENDER GRAPH         MB    {}\n", header_unit));

    for n in &summary.nodes {
        let indent = "  ".repeat(n.depth.max(1) as usize - if n.depth == 0 { 0 } else { 1 });
        let branch = if n.depth == 0 { "" } else { "+- " };
        let mut tags = String::new();
        if n.is_present {
            tags.push_str(" *");
        }
        if n.viewer_enabled {
            tags.push_str(" v");
        }
        let left = format!("{indent}{branch}{}{}", n.id, tags);
        let mb = n.bytes as f32 / (1024.0 * 1024.0);
        let ms = if has_gpu {
            n.gpu_ms.unwrap_or(0.0)
        } else {
            n.dispatch_ms
        };
        s.push_str(&format!("{:<21}{:>5.2} {:>6.2}\n", left, mb, ms));
    }

    let total_mib = summary.total_bytes as f32 / (1024.0 * 1024.0);
    s.push_str(&format!(
        "total: {:.2} MiB / {} passes\n",
        total_mib, passes
    ));
    s.push_str(&format!("GPU:     {}\n", hud::gpu_label(adapter_info)));
    s.push_str(&format!(
        "Type:    {}\n",
        hud::device_type_label(adapter_info.device_type),
    ));
    s.push_str(&format!(
        "Backend: {}  CPU: {:.2} ms\n",
        hud::backend_label(adapter_info.backend),
        cpu_ms
    ));
    s.push_str(&format!("FPS:     {:.1}", fps));
    s
}

pub fn run() {
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("event loop");
    let proxy = event_loop.create_proxy();

    #[cfg(target_arch = "wasm32")]
    let lock = parse_test_lock_from_url();
    #[cfg(not(target_arch = "wasm32"))]
    let lock = TestLock::default();

    if lock.time.is_some() || lock.mouse_uv.is_some() {
        log::info!(
            "test lock: time={:?}, mouse_uv={:?}",
            lock.time,
            lock.mouse_uv
        );
    }

    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut app = App {
        proxy: Some(proxy),
        state: None,
        lock,
    };

    #[cfg(not(target_arch = "wasm32"))]
    event_loop.run_app(&mut app).expect("run_app");

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn web_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let _ = console_log::init_with_level(log::Level::Info);
    run();
}

/// Locked inputs for a single offscreen frame. `time` is the value the
/// fragment shader sees in `u.time`; `mouse_uv` is the cursor in
/// normalized canvas coordinates (top-left origin, matches the `mx`/`my`
/// URL params on web).
#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub time: f32,
    pub mouse_uv: [f32; 2],
}

/// Render exactly one frame to an offscreen `Rgba8UnormSrgb` texture and
/// return the pixels as tightly-packed RGBA8 bytes (length = width * height * 4).
/// Used by the native test harness to mirror the deterministic-frame check
/// the browser test does. No surface, no event loop — just the pipeline.
#[cfg(not(target_arch = "wasm32"))]
pub async fn render_offscreen(cfg: RenderConfig) -> Vec<u8> {
    assert!(cfg.width > 0 && cfg.height > 0);

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
        .expect("offscreen: no adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("offscreen-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .await
        .expect("offscreen: request_device");

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut graph = RenderGraph::from_toml(&device, format, DEFAULT_GRAPH_TOML)
        .unwrap_or_else(|e| panic!("config/graph.toml: {e}"));
    graph.resize(&device, cfg.width, cfg.height);

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen-target"),
        size: wgpu::Extent3d {
            width: cfg.width,
            height: cfg.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // wgpu's copy_texture_to_buffer requires bytes_per_row to be a multiple
    // of 256. We pad in the staging buffer and strip the padding on readback.
    const BYTES_PER_PIXEL: u32 = 4;
    let unpadded_bpr = cfg.width * BYTES_PER_PIXEL;
    let padded_bpr =
        (unpadded_bpr + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1) / wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen-readback"),
        size: (padded_bpr * cfg.height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen-encoder"),
    });
    {
        let mut ctx = NodeContext {
            encoder: &mut encoder,
            queue: &queue,
            output: &view,
            width: cfg.width,
            height: cfg.height,
            time: cfg.time,
            mouse: [
                cfg.mouse_uv[0] * cfg.width as f32,
                cfg.mouse_uv[1] * cfg.height as f32,
            ],
        };
        graph.render(&mut ctx);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(cfg.height),
            },
        },
        wgpu::Extent3d {
            width: cfg.width,
            height: cfg.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().expect("buffer map failed");

    let padded = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded_bpr * cfg.height) as usize);
    for row in 0..cfg.height {
        let start = (row * padded_bpr) as usize;
        out.extend_from_slice(&padded[start..start + unpadded_bpr as usize]);
    }
    drop(padded);
    staging.unmap();
    out
}
