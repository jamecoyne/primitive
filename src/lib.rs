use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    mouse: [f32; 2],
    time: f32,
    // pad to 32 bytes — std140 requires the block size to be a multiple of 16
    _pad: [f32; 3],
}

/// Test-mode overrides parsed from URL params on web. None on native.
/// When `time` is Some, the time uniform is frozen at that value. When
/// `mouse_uv` is Some, the cursor is pinned and CursorMoved events are
/// ignored — used by the screenshot harness to render deterministic frames.
#[derive(Clone, Copy, Default)]
struct TestLock {
    time: Option<f32>,
    mouse_uv: Option<[f32; 2]>,
}

/// Pipeline + uniform buffer + bind group for the mandelbrot. Holds nothing
/// surface-specific so it can be reused by the offscreen render path used in
/// native tests. Created once per device.
struct Renderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

fn create_renderer(device: &wgpu::Device, format: wgpu::TextureFormat) -> Renderer {
    // GLSL → naga → backend SPIR-V/MSL/WGSL/GLSL ES.
    let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mandelbrot.vert"),
        source: wgpu::ShaderSource::Glsl {
            shader: include_str!("../shaders/mandelbrot.vert").into(),
            stage: wgpu::naga::ShaderStage::Vertex,
            defines: Default::default(),
        },
    });
    let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mandelbrot.frag"),
        source: wgpu::ShaderSource::Glsl {
            shader: include_str!("../shaders/mandelbrot.frag").into(),
            stage: wgpu::naga::ShaderStage::Fragment,
            defines: Default::default(),
        },
    });

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bind-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind-group"),
        layout: &bind_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline-layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mandelbrot-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vs,
            entry_point: Some("main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs,
            entry_point: Some("main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    Renderer {
        pipeline,
        uniform_buffer,
        bind_group,
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    // Format used for the render-pass color attachment. May differ from
    // config.format: on web the WebGPU canvas only accepts non-sRGB
    // storage formats, so we render through an sRGB view of the canvas
    // texture to keep the linear→sRGB encoding consistent with native.
    view_format: wgpu::TextureFormat,
    start: web_time::Instant,
    mouse: [f32; 2],
    lock: TestLock,
}

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

        log::info!("adapter: {:?}", adapter.get_info());

        let required_limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits::downlevel_webgl2_defaults()
        } else {
            wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
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

        let renderer = create_renderer(&device, view_format);

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
            renderer,
            view_format,
            start: web_time::Instant::now(),
            mouse,
            lock,
        }
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        // When the cursor is pinned via test lock, keep it pinned in uv space
        // even as the canvas resizes — otherwise locked-frame screenshots
        // would drift after the first ResizeObserver event on web.
        if let Some([u, v]) = self.lock.mouse_uv {
            self.mouse = [u * size.width as f32, v * size.height as f32];
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });

        let time = self
            .lock
            .time
            .unwrap_or_else(|| self.start.elapsed().as_secs_f32());
        let uniforms = Uniforms {
            resolution: [self.config.width as f32, self.config.height as f32],
            mouse: self.mouse,
            time,
            _pad: [0.0; 3],
        };
        self.queue
            .write_buffer(&self.renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.renderer.pipeline);
            pass.set_bind_group(0, &self.renderer.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
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
            .with_title("Mandelbrot — wgpu")
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
                if state.lock.mouse_uv.is_none() {
                    state.mouse = [position.x as f32, position.y as f32];
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
    let renderer = create_renderer(&device, format);

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

    let uniforms = Uniforms {
        resolution: [cfg.width as f32, cfg.height as f32],
        mouse: [
            cfg.mouse_uv[0] * cfg.width as f32,
            cfg.mouse_uv[1] * cfg.height as f32,
        ],
        time: cfg.time,
        _pad: [0.0; 3],
    };
    queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

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
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("offscreen-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&renderer.pipeline);
        pass.set_bind_group(0, &renderer.bind_group, &[]);
        pass.draw(0..3, 0..1);
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
