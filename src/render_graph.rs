//! Render graph — list of nodes loaded from `config/graph.toml` and walked
//! in topological order. Non-present nodes write to intermediate
//! `Rgba8UnormSrgb` textures; downstream nodes sample those as fragment
//! shader inputs. Designed roughly after bevy 0.18's `bevy_render::render_graph`
//! (Node trait, slot/edge model, topological execute) but stripped of
//! bevy-isms (no ECS, no sub-graphs, no view-driven dispatch).

use std::collections::{HashMap, HashSet};

use bytemuck::{Pod, Zeroable};
use serde::Deserialize;

/// Per-frame globals shared by every fragment-shader node.
///
/// std140 alignment: vec2 fields are 8-byte-aligned; the block size must be
/// a multiple of 16. Total size here is 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct Uniforms {
    pub resolution: [f32; 2],
    pub mouse: [f32; 2],
    pub time: f32,
    pub _pad: [f32; 3],
}

/// Inputs handed to a node every frame. Equivalent to bevy's
/// `RenderGraphContext` + the read-only chunks of `RenderContext`.
pub struct NodeContext<'a> {
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub queue: &'a wgpu::Queue,
    /// Color-attachment view this node should render into. For a `present`
    /// node this is the swapchain (or offscreen target); for non-present
    /// nodes the graph passes the appropriate intermediate view.
    pub output: &'a wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub time: f32,
    /// Cursor position in physical pixels (winit y-down).
    pub mouse: [f32; 2],
}

/// A render-graph node. Implementations own their pipeline + per-node
/// uniform buffer + bind group; `record` issues whatever GPU work is needed
/// to fill `ctx.output`. `set_inputs` is called once at graph build time
/// (and again on resize) with views for every declared input slot, in the
/// order they appear in `NodeConfig::inputs`.
///
/// `record` takes an `Option<RenderPassTimestampWrites>`: when the device
/// supports `Features::TIMESTAMP_QUERY` and the perf monitor is active,
/// the graph allocates two query indices per pass and threads them in.
/// Implementations should plumb the option directly into the
/// `RenderPassDescriptor::timestamp_writes` field; pass `None` and the
/// pass runs untimed.
///
/// No `Send + Sync` bound — wgpu's pipeline/buffer/bind-group handles are
/// `!Send + !Sync` on wasm32 (they wrap JS handles), and the graph is only
/// ever walked from the render thread.
pub trait Node {
    fn record(
        &self,
        ctx: &mut NodeContext<'_>,
        ts_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    );
    /// Build (or rebuild) the bind group with the given input texture views.
    /// `views.len()` always matches the node's declared input count.
    fn set_inputs(&mut self, device: &wgpu::Device, views: &[&wgpu::TextureView]);
}

// ---------------------------------------------------------------------------
// GlslNode — vert/frag pair with N input texture slots
// ---------------------------------------------------------------------------

/// Generic fragment-shader node. The bind layout is:
///
///   set=0, binding=0  : Uniforms (always)
///   set=0, binding=1  : sampler (when input_count > 0)
///   set=0, binding=2..2+N : input texture views
///
/// The fragment shader is expected to declare matching bindings; for
/// non-input nodes (e.g. mandelbrot) it only declares the uniform.
pub struct GlslNode {
    label: String,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_layout: wgpu::BindGroupLayout,
    sampler: Option<wgpu::Sampler>,
    bind_group: Option<wgpu::BindGroup>,
    input_count: usize,
}

impl GlslNode {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        vert_src: &str,
        frag_src: &str,
        label: &str,
        input_count: usize,
    ) -> Self {
        let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}.vert")),
            source: wgpu::ShaderSource::Glsl {
                shader: vert_src.into(),
                stage: wgpu::naga::ShaderStage::Vertex,
                defines: Default::default(),
            },
        });
        let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}.frag")),
            source: wgpu::ShaderSource::Glsl {
                shader: frag_src.into(),
                stage: wgpu::naga::ShaderStage::Fragment,
                defines: Default::default(),
            },
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}:uniforms")),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind layout — uniform always, sampler + textures only if N>0.
        let mut layout_entries: Vec<wgpu::BindGroupLayoutEntry> = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        let sampler = if input_count > 0 {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            });
            for i in 0..input_count {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 2 + i as u32,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
            }
            // Nearest sampling: 1:1 fullscreen quad makes filtering moot,
            // and this avoids any cross-driver filtering ambiguity that
            // would break the cross-platform pixel-perfect test.
            Some(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(&format!("{label}:sampler")),
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            }))
        } else {
            None
        };

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}:bind-layout")),
            entries: &layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}:pipeline-layout")),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{label}:pipeline")),
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

        Self {
            label: label.to_string(),
            pipeline,
            uniform_buffer,
            bind_layout,
            sampler,
            bind_group: None,
            input_count,
        }
    }
}

impl Node for GlslNode {
    fn record(
        &self,
        ctx: &mut NodeContext<'_>,
        ts_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let bind_group = self
            .bind_group
            .as_ref()
            .expect("GlslNode bind_group not set — graph.resize must run before render");

        let uniforms = Uniforms {
            resolution: [ctx.width as f32, ctx.height as f32],
            mouse: ctx.mouse,
            time: ctx.time,
            _pad: [0.0; 3],
        };
        ctx.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let pass_label = format!("{}:pass", self.label);
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&pass_label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.output,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: ts_writes,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn set_inputs(&mut self, device: &wgpu::Device, views: &[&wgpu::TextureView]) {
        assert_eq!(
            views.len(),
            self.input_count,
            "GlslNode {:?}: expected {} input view(s), got {}",
            self.label,
            self.input_count,
            views.len(),
        );

        let mut entries: Vec<wgpu::BindGroupEntry> = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_entire_binding(),
            },
        ];
        if self.input_count > 0 {
            let sampler = self.sampler.as_ref().expect("sampler created when input_count>0");
            entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
            for (i, view) in views.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: 2 + i as u32,
                    resource: wgpu::BindingResource::TextureView(view),
                });
            }
        }

        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}:bind-group", self.label)),
            layout: &self.bind_layout,
            entries: &entries,
        }));
    }
}

// ---------------------------------------------------------------------------
// Config (TOML)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct GraphConfig {
    /// Top-level `[out]` block — declares which node's output is rendered
    /// to the swapchain. Required.
    pub out: OutConfig,
    #[serde(rename = "node", default)]
    pub nodes: Vec<NodeConfig>,
}

/// Top-level `[out]` table. Currently a single field; kept as a struct
/// so future fields (e.g. tone mapping, color space, blend mode) can land
/// without breaking the TOML schema.
#[derive(Debug, Deserialize)]
pub struct OutConfig {
    /// Id of the node whose output gets blitted to the swapchain.
    pub input: String,
}

/// Per-node viewer settings. Modeled after TouchDesigner's per-operator
/// "viewer active" toggle and viewer resolution: when `enabled`, the
/// node's full intermediate is downsampled into a thumbnail-sized
/// texture every frame and overlaid as a PiP at a configurable pixel
/// offset on the swapchain.
///
/// TOML form (inline table per node):
///
/// ```toml
/// viewer = { enabled = true, resolution = [128, 96], position = [16, 16] }
/// ```
///
/// All fields are optional; an absent `viewer` table defaults to
/// `enabled = false`, in which case no viewer texture is allocated.
#[derive(Debug, Clone, Deserialize)]
pub struct ViewerConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_viewer_resolution")]
    pub resolution: [u32; 2],
    /// Top-left corner of the PiP thumbnail in physical pixels from the
    /// canvas's top-left. Default `[16, 16]`. Two viewers with the same
    /// position will overlap — the user is responsible for assigning
    /// non-overlapping offsets. Thumbnails that would overflow the
    /// canvas (`x + width > canvas_w` or `y + height > canvas_h`) are
    /// silently skipped for that frame so a misconfigured offset
    /// doesn't blow up the render.
    #[serde(default = "default_viewer_position")]
    pub position: [u32; 2],
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            resolution: default_viewer_resolution(),
            position: default_viewer_position(),
        }
    }
}

fn default_viewer_resolution() -> [u32; 2] {
    [128, 96]
}

fn default_viewer_position() -> [u32; 2] {
    [16, 16]
}

#[derive(Debug, Deserialize)]
pub struct NodeConfig {
    pub id: String,
    pub kind: String,
    /// Required when `kind = "glsl"` — full GLSL source for the vertex stage.
    #[serde(default)]
    pub vert: Option<String>,
    /// Required when `kind = "glsl"` — full GLSL source for the fragment stage.
    #[serde(default)]
    pub frag: Option<String>,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub viewer: ViewerConfig,
}

/// Parse a TOML graph definition. Returns a structured error on bad input
/// rather than panicking.
pub fn parse_config(toml_src: &str) -> Result<GraphConfig, String> {
    toml::from_str(toml_src).map_err(|e| format!("graph.toml parse error: {e}"))
}

/// Compile-time embedded default graph definition.
pub const DEFAULT_GRAPH_TOML: &str = include_str!("../config/graph.toml");

// ---------------------------------------------------------------------------
// Graph — chaining, intermediates, topological execute
// ---------------------------------------------------------------------------

/// Format used for every intermediate texture. Same sRGB encoding on
/// every render-pass write keeps cross-platform parity at the swapchain
/// boundary: sampling the intermediate decodes back to linear, writing
/// to the swapchain re-encodes — round-trip is exact at 8-bit precision.
const INTERMEDIATE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

// ---------------------------------------------------------------------------
// Blitter — generic fullscreen-quad sample → write pipeline. Used twice by
// the graph: once with the swapchain's `present_format` for the final
// present-blit, and once with `INTERMEDIATE_FORMAT` for downsampling each
// node's full intermediate into its viewer texture.
// ---------------------------------------------------------------------------

const BLIT_VERT: &str = "#version 450
layout(location = 0) out vec2 v_uv;
void main() {
    vec2 pos = vec2(
        float((gl_VertexIndex & 1) * 4) - 1.0,
        float((gl_VertexIndex & 2) * 2) - 1.0
    );
    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
";

const BLIT_FRAG: &str = "#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;
layout(set = 0, binding = 0) uniform sampler   s_in;
layout(set = 0, binding = 1) uniform texture2D t_in;
void main() {
    // Texture y=0 is the top of the image; v_uv y=0 is the bottom of the
    // viewport. Flip y so the blitted frame stays upright.
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    o_color = texture(sampler2D(t_in, s_in), uv);
}
";

struct Blitter {
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    bind_layout: wgpu::BindGroupLayout,
}

impl Blitter {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, label: &str) -> Self {
        let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}.vert")),
            source: wgpu::ShaderSource::Glsl {
                shader: BLIT_VERT.into(),
                stage: wgpu::naga::ShaderStage::Vertex,
                defines: Default::default(),
            },
        });
        let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}.frag")),
            source: wgpu::ShaderSource::Glsl {
                shader: BLIT_FRAG.into(),
                stage: wgpu::naga::ShaderStage::Fragment,
                defines: Default::default(),
            },
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{label}:sampler")),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}:bind-layout")),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}:pipeline-layout")),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{label}:pipeline")),
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
                    format: target_format,
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
        Self {
            pipeline,
            sampler,
            bind_layout,
        }
    }

    fn make_bind_group(
        &self,
        device: &wgpu::Device,
        source_view: &wgpu::TextureView,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
            ],
        })
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        label: &str,
        ts_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: ts_writes,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Variant of `record` that draws into a sub-rect of the target without
    /// clearing the rest. Used to lay PiP viewer thumbnails on top of the
    /// already-blitted swapchain. `viewport` is `[x, y, width, height]` in
    /// physical pixels with the framebuffer's top-left as origin.
    fn record_overlay(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        viewport: [f32; 4],
        label: &str,
        ts_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Preserve everything outside the viewport (the present
                    // blit just wrote it).
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: ts_writes,
        });
        pass.set_viewport(
            viewport[0], viewport[1], viewport[2], viewport[3], 0.0, 1.0,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

// ---------------------------------------------------------------------------
// SolidFill — fullscreen-triangle pipeline whose fragment outputs a single
// hardcoded color and takes no bindings. Used for the per-viewer border
// pass; expand a viewer's rect, fill it with this, then blit the
// thumbnail on top to leave a border ring.
// ---------------------------------------------------------------------------

const SOLID_FILL_VERT: &str = "#version 450
void main() {
    vec2 pos = vec2(
        float((gl_VertexIndex & 1) * 4) - 1.0,
        float((gl_VertexIndex & 2) * 2) - 1.0
    );
    gl_Position = vec4(pos, 0.0, 1.0);
}
";

const SOLID_FILL_FRAG: &str = "#version 450
layout(location = 0) out vec4 o_color;
void main() {
    // Near-black with a slight cool tint — readable border around any
    // thumbnail content (the magenta source and the inverted thumbnail
    // both have light-coloured surrounds, so a dark border stands out).
    o_color = vec4(0.07, 0.07, 0.09, 1.0);
}
";

struct SolidFill {
    pipeline: wgpu::RenderPipeline,
}

impl SolidFill {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, label: &str) -> Self {
        let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}.vert")),
            source: wgpu::ShaderSource::Glsl {
                shader: SOLID_FILL_VERT.into(),
                stage: wgpu::naga::ShaderStage::Vertex,
                defines: Default::default(),
            },
        });
        let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}.frag")),
            source: wgpu::ShaderSource::Glsl {
                shader: SOLID_FILL_FRAG.into(),
                stage: wgpu::naga::ShaderStage::Fragment,
                defines: Default::default(),
            },
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label}:pipeline-layout")),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{label}:pipeline")),
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
                    format: target_format,
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
        Self { pipeline }
    }

    /// Fill `viewport` with the constant fragment color. Uses
    /// `LoadOp::Load` so pixels outside the viewport are preserved.
    fn record_overlay(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        viewport: [f32; 4],
        label: &str,
        ts_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: ts_writes,
        });
        pass.set_viewport(viewport[0], viewport[1], viewport[2], viewport[3], 0.0, 1.0);
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
    }
}

#[derive(Debug)]
pub enum GraphError {
    Config(String),
    UnknownNodeKind { id: String, kind: String },
    /// `kind = "glsl"` requires both `vert` and `frag` fields with full
    /// GLSL source. Names which one is missing.
    MissingShaderStage { id: String, stage: &'static str },
    DuplicateNodeId { id: String },
    /// `[out].input` references a node id that wasn't declared.
    UnknownOutInput { input: String },
    /// The named upstream id wasn't declared in the config.
    UnknownInput { node: String, input: String },
    /// The graph contains a cycle that includes the named node.
    CycleInvolving { node: String },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::Config(s) => write!(f, "{s}"),
            GraphError::UnknownNodeKind { id, kind } => {
                write!(f, "node {id:?} has unknown kind {kind:?}")
            }
            GraphError::MissingShaderStage { id, stage } => {
                write!(
                    f,
                    "node {id:?} (kind = \"glsl\") is missing required `{stage}` field"
                )
            }
            GraphError::DuplicateNodeId { id } => {
                write!(f, "two or more nodes share id {id:?}")
            }
            GraphError::UnknownOutInput { input } => {
                write!(f, "[out].input references unknown node {input:?}")
            }
            GraphError::UnknownInput { node, input } => {
                write!(f, "node {node:?} references unknown input {input:?}")
            }
            GraphError::CycleInvolving { node } => {
                write!(f, "cycle detected: node {node:?} is part of a cycle")
            }
        }
    }
}

impl std::error::Error for GraphError {}

struct NodeEntry {
    id: String,
    /// Upstream node indices in the order declared in `NodeConfig::inputs`.
    inputs: Vec<usize>,
    impl_: Box<dyn Node>,
    /// Rolling buffer of CPU dispatch times (record() durations) in
    /// seconds. Note: this is encoder build-up CPU time, NOT real GPU
    /// cook time — the latter requires `Features::TIMESTAMP_QUERY` plus
    /// ring-buffer async readback and is left as a follow-up.
    dispatch_times: std::collections::VecDeque<f32>,
}

/// Window length for the dispatch-time ring on each node. Matches
/// `lib::FRAME_WINDOW`.
const NODE_TIMING_WINDOW: usize = 60;

pub struct RenderGraph {
    nodes: Vec<NodeEntry>,
    /// Node indices in topological order: every node's `inputs` precede it.
    schedule: Vec<usize>,
    /// Index in `nodes` of the unique node with `present = true`. Its
    /// intermediate is what `blit` copies to the swapchain.
    present_index: usize,
    /// One slot per node, in the same order as `nodes`. Every node has
    /// its own intermediate so any node can be both consumed by another
    /// node and "presented". Reallocated on `resize`; `None` only before
    /// the first `resize` call.
    intermediates: Vec<Option<Intermediate>>,
    /// Per-node viewer settings carried forward from the TOML. Parallel
    /// to `nodes`.
    viewer_configs: Vec<ViewerConfig>,
    /// Per-node viewer state. `Some` only when the corresponding node has
    /// `viewer.enabled = true`. Reallocated on `resize`.
    viewers: Vec<Option<Viewer>>,
    /// Blit pipeline that targets the swapchain (`present_format`).
    present_blit: Blitter,
    /// Bind group bound to `intermediates[present_index]`. Re-built on resize.
    present_bind_group: Option<wgpu::BindGroup>,
    /// Blit pipeline that targets `INTERMEDIATE_FORMAT`. Shared across
    /// every viewer-enabled node.
    viewer_blit: Blitter,
    /// Solid-color fill used for the thick border drawn behind each
    /// viewer thumbnail. Targets `present_format`.
    border_fill: SolidFill,
    /// Optional GPU timestamps. `None` when the device didn't expose
    /// `Features::TIMESTAMP_QUERY` (web fallback, older drivers).
    /// Enabled by `enable_perf_monitor` after build.
    perf: Option<PerfMonitor>,
    /// Pre-built per-pass labels so `next_writes` doesn't re-`format!`
    /// every frame. One per node for each of the four pass sites.
    schedule_labels: Vec<String>,
    viewer_downsample_labels: Vec<String>,
    viewer_border_labels: Vec<String>,
    viewer_pip_labels: Vec<String>,
    width: u32,
    height: u32,
}

/// Pixel thickness of the border drawn around each viewer thumbnail.
const VIEWER_BORDER_PX: u32 = 6;

/// Per-node viewer state — a thumbnail-sized texture filled every frame
/// by downsampling the node's full intermediate. Two bind groups because
/// the two passes that read/write the viewer use blitters with different
/// target formats (and therefore different bind-group layouts):
///
///   - `downsample_bg` is bound to the source intermediate; consumed by
///     `viewer_blit` (target = INTERMEDIATE_FORMAT) to fill `view`.
///   - `pip_bg` is bound to `view` itself; consumed by `present_blit`
///     (target = present_format) to overlay the thumbnail on the
///     swapchain.
struct Viewer {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    downsample_bg: wgpu::BindGroup,
    pip_bg: wgpu::BindGroup,
}

/// Read-only handle to one viewer's texture, returned from
/// `RenderGraph::viewer_textures()` so a future preview UI can sample it.
pub struct ViewerSlot<'a> {
    pub node_id: &'a str,
    pub view: &'a wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

/// Counters returned by `RenderGraph::render` so the caller (HUD, perf
/// logging) can show what the graph just did.
#[derive(Default, Debug, Clone, Copy)]
pub struct FrameStats {
    /// Total number of `begin_render_pass` calls the graph submitted —
    /// schedule walk + viewer downsamples + present blit + PiP overlays.
    pub passes: u32,
}

/// Snapshot of the graph's static structure + recent timings, returned
/// from `RenderGraph::summary()`. Drives the perf-monitor HUD layout.
#[derive(Debug, Clone)]
pub struct GraphSummary {
    pub nodes: Vec<NodeSummary>,
    /// Sum of `bytes` across every entry, plus zero for any node that
    /// has neither an intermediate nor a viewer allocated.
    pub total_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct NodeSummary {
    pub id: String,
    /// Distance from the deepest leaf — 0 for nodes with no inputs,
    /// `max(input.depth) + 1` otherwise. Drives the tree indent.
    pub depth: u32,
    /// Width × height × 4 of (intermediate + viewer) for this node.
    pub bytes: u64,
    /// Whether this node is `[out].input`.
    pub is_present: bool,
    /// Whether `viewer.enabled = true` in the TOML.
    pub viewer_enabled: bool,
    /// Rolling-average CPU time spent inside this node's `record()`
    /// call, in milliseconds. Reflects encoder dispatch only.
    pub dispatch_ms: f32,
    /// Rolling-average GPU cook time for this node's schedule pass, in
    /// milliseconds — measured via `Features::TIMESTAMP_QUERY` when the
    /// device supports it. `None` when the perf monitor isn't running
    /// or hasn't received a sample yet.
    pub gpu_ms: Option<f32>,
}

struct Intermediate {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl RenderGraph {
    /// Build the graph from a parsed config. Topologically sorts and
    /// instantiates every node. Intermediate textures are not allocated
    /// here — call `resize(device, w, h)` (which `State::new` already does)
    /// before the first render.
    pub fn build(
        device: &wgpu::Device,
        present_format: wgpu::TextureFormat,
        cfg: &GraphConfig,
    ) -> Result<Self, GraphError> {
        // 1. Unique ids.
        let mut id_to_idx: HashMap<&str, usize> = HashMap::new();
        for (i, n) in cfg.nodes.iter().enumerate() {
            if id_to_idx.insert(n.id.as_str(), i).is_some() {
                return Err(GraphError::DuplicateNodeId { id: n.id.clone() });
            }
        }

        // 2. Resolve `[out].input` to the index of the node whose output
        //    gets blitted to the swapchain.
        let present_index = *id_to_idx
            .get(cfg.out.input.as_str())
            .ok_or_else(|| GraphError::UnknownOutInput {
                input: cfg.out.input.clone(),
            })?;

        // 3. Resolve inputs into upstream indices.
        let mut input_indices: Vec<Vec<usize>> = Vec::with_capacity(cfg.nodes.len());
        for n in &cfg.nodes {
            let mut idxs = Vec::with_capacity(n.inputs.len());
            for input_id in &n.inputs {
                let &upstream = id_to_idx.get(input_id.as_str()).ok_or_else(|| {
                    GraphError::UnknownInput {
                        node: n.id.clone(),
                        input: input_id.clone(),
                    }
                })?;
                idxs.push(upstream);
            }
            input_indices.push(idxs);
        }

        // 4. Topological sort (DFS post-order with cycle detection).
        let schedule = topo_sort(&cfg.nodes, &input_indices)?;

        // 5. Instantiate nodes. Every node — present or not — renders into
        //    its own intermediate texture; an internal blit pass copies the
        //    present node's intermediate to the swapchain at the end.
        let mut nodes: Vec<NodeEntry> = Vec::with_capacity(cfg.nodes.len());
        for (i, n) in cfg.nodes.iter().enumerate() {
            let _ = i; // (no per-node format branch any more)
            let format = INTERMEDIATE_FORMAT;
            let input_count = input_indices[i].len();
            let impl_: Box<dyn Node> = match n.kind.as_str() {
                "glsl" => {
                    let vert_src = n
                        .vert
                        .as_deref()
                        .ok_or_else(|| GraphError::MissingShaderStage {
                            id: n.id.clone(),
                            stage: "vert",
                        })?;
                    let frag_src = n
                        .frag
                        .as_deref()
                        .ok_or_else(|| GraphError::MissingShaderStage {
                            id: n.id.clone(),
                            stage: "frag",
                        })?;
                    Box::new(GlslNode::new(
                        device,
                        format,
                        vert_src,
                        frag_src,
                        &n.id,
                        input_count,
                    ))
                }
                other => {
                    return Err(GraphError::UnknownNodeKind {
                        id: n.id.clone(),
                        kind: other.to_string(),
                    })
                }
            };
            nodes.push(NodeEntry {
                id: n.id.clone(),
                inputs: input_indices[i].clone(),
                impl_,
                dispatch_times: std::collections::VecDeque::with_capacity(NODE_TIMING_WINDOW),
            });
        }

        let intermediates = (0..nodes.len()).map(|_| None).collect();
        let viewers = (0..nodes.len()).map(|_| None).collect();
        let viewer_configs: Vec<ViewerConfig> =
            cfg.nodes.iter().map(|n| n.viewer.clone()).collect();
        let present_blit = Blitter::new(device, present_format, "present-blit");
        let viewer_blit = Blitter::new(device, INTERMEDIATE_FORMAT, "viewer-blit");
        let border_fill = SolidFill::new(device, present_format, "viewer-border");

        log::info!(
            "render graph built: {} node(s), schedule = {:?}, present = {:?}, viewers = {:?}",
            nodes.len(),
            schedule
                .iter()
                .map(|&i| nodes[i].id.as_str())
                .collect::<Vec<_>>(),
            nodes[present_index].id,
            viewer_configs
                .iter()
                .enumerate()
                .filter(|(_, v)| v.enabled)
                .map(|(i, v)| (nodes[i].id.as_str(), v.resolution))
                .collect::<Vec<_>>(),
        );

        // Pre-build per-pass labels so PerfMonitor::next_writes doesn't
        // reformat strings on every frame.
        let schedule_labels = nodes
            .iter()
            .map(|e| format!("schedule:{}", e.id))
            .collect();
        let viewer_downsample_labels = nodes
            .iter()
            .map(|e| format!("viewer-down:{}", e.id))
            .collect();
        let viewer_border_labels = nodes
            .iter()
            .map(|e| format!("viewer-border:{}", e.id))
            .collect();
        let viewer_pip_labels = nodes
            .iter()
            .map(|e| format!("viewer-pip:{}", e.id))
            .collect();

        Ok(Self {
            nodes,
            schedule,
            present_index,
            intermediates,
            viewer_configs,
            viewers,
            present_blit,
            present_bind_group: None,
            viewer_blit,
            border_fill,
            perf: None,
            schedule_labels,
            viewer_downsample_labels,
            viewer_border_labels,
            viewer_pip_labels,
            width: 0,
            height: 0,
        })
    }

    /// Enable per-pass GPU cook timing. Must be called after `build` and
    /// before the first `render` if you want the data flowing. Silently
    /// no-ops on devices that don't expose `Features::TIMESTAMP_QUERY`.
    pub fn enable_perf_monitor(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return;
        }
        let period = queue.get_timestamp_period();
        self.perf = Some(PerfMonitor::new(device, period));
        log::info!(
            "perf monitor: enabled, timestamp period = {:.3} ns/tick",
            period
        );
    }

    /// Drives async readbacks for the perf monitor. Must be called once
    /// per frame, **after** `queue.submit`. No-op when the perf monitor
    /// isn't enabled.
    pub fn poll_perf(&mut self, device: &wgpu::Device) {
        if let Some(p) = &mut self.perf {
            p.after_submit(device);
        }
    }

    /// Convenience: parse + build from a raw TOML string.
    pub fn from_toml(
        device: &wgpu::Device,
        present_format: wgpu::TextureFormat,
        toml_src: &str,
    ) -> Result<Self, GraphError> {
        let cfg = parse_config(toml_src).map_err(GraphError::Config)?;
        Self::build(device, present_format, &cfg)
    }

    /// (Re)allocate intermediate textures, rebuild every node's bind
    /// group, and re-point the blit pass at the present node's view.
    /// Call once after `build`, and again whenever the canvas size changes.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.width = width;
        self.height = height;

        // (Re)allocate an intermediate for every node — including the
        // present node, since the blit pass samples it.
        for (i, entry) in self.nodes.iter().enumerate() {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("{}:intermediate", entry.id)),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: INTERMEDIATE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.intermediates[i] = Some(Intermediate { texture, view });
        }

        // Rebuild every node's bind group with fresh upstream views.
        for i in 0..self.nodes.len() {
            let inputs = self.nodes[i].inputs.clone();
            let views: Vec<&wgpu::TextureView> = inputs
                .iter()
                .map(|&j| {
                    &self.intermediates[j]
                        .as_ref()
                        .expect("upstream intermediate must be allocated")
                        .view
                })
                .collect();
            self.nodes[i].impl_.set_inputs(device, &views);
        }

        // Point the present blit at the present node's intermediate.
        let present_view = &self.intermediates[self.present_index]
            .as_ref()
            .expect("present intermediate must be allocated")
            .view;
        self.present_bind_group = Some(self.present_blit.make_bind_group(
            device,
            present_view,
            "present-blit:bind-group",
        ));

        // (Re)allocate viewer textures + their blit bind groups for every
        // node where `viewer.enabled = true`. Disabled nodes get None.
        for (i, entry) in self.nodes.iter().enumerate() {
            let cfg = &self.viewer_configs[i];
            if !cfg.enabled {
                self.viewers[i] = None;
                continue;
            }
            let vw = cfg.resolution[0].max(1);
            let vh = cfg.resolution[1].max(1);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("{}:viewer", entry.id)),
                size: wgpu::Extent3d {
                    width: vw,
                    height: vh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: INTERMEDIATE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let source_view = &self.intermediates[i]
                .as_ref()
                .expect("intermediate just allocated above")
                .view;
            let downsample_bg = self.viewer_blit.make_bind_group(
                device,
                source_view,
                &format!("{}:viewer:downsample-bg", entry.id),
            );
            let pip_bg = self.present_blit.make_bind_group(
                device,
                &view,
                &format!("{}:viewer:pip-bg", entry.id),
            );
            self.viewers[i] = Some(Viewer {
                texture,
                view,
                width: vw,
                height: vh,
                downsample_bg,
                pip_bg,
            });
        }
    }

    /// Walk the schedule (every node renders into its own intermediate),
    /// downsample any viewer-enabled nodes into their viewer textures,
    /// then run the present blit to copy the present node's intermediate
    /// into `ctx.output`. Returns counters describing the work submitted.
    pub fn render(&mut self, ctx: &mut NodeContext<'_>) -> FrameStats {
        if let Some(p) = &mut self.perf {
            p.begin_frame();
        }

        let mut passes: u32 = 0;
        // Schedule walk. Split-borrow: `self.perf` and `self.nodes` are
        // separate fields, so we can mut-borrow perf for next_writes()
        // while still hitting nodes[i] for record() and dispatch_times.
        let schedule = self.schedule.clone();
        for node_idx in schedule {
            let output = &self.intermediates[node_idx]
                .as_ref()
                .expect("intermediate must be allocated by resize before render")
                .view;
            let mut sub_ctx = NodeContext {
                encoder: &mut *ctx.encoder,
                queue: ctx.queue,
                output,
                width: ctx.width,
                height: ctx.height,
                time: ctx.time,
                mouse: ctx.mouse,
            };
            let writes = self
                .perf
                .as_mut()
                .and_then(|p| p.next_writes(&self.schedule_labels[node_idx]));
            let dispatch_start = web_time::Instant::now();
            self.nodes[node_idx].impl_.record(&mut sub_ctx, writes);
            let dt = dispatch_start.elapsed().as_secs_f32();
            let entry = &mut self.nodes[node_idx];
            if entry.dispatch_times.len() == NODE_TIMING_WINDOW {
                entry.dispatch_times.pop_front();
            }
            entry.dispatch_times.push_back(dt);
            passes += 1;
        }

        // Viewer downsamples — independent of present, can run in any
        // order relative to it. Must precede the PiP overlays below since
        // those sample the textures these passes write.
        for i in 0..self.viewers.len() {
            let Some(v) = &self.viewers[i] else { continue };
            let label = format!("{}:viewer:downsample-pass", self.nodes[i].id);
            let writes = self
                .perf
                .as_mut()
                .and_then(|p| p.next_writes(&self.viewer_downsample_labels[i]));
            self.viewer_blit
                .record(ctx.encoder, &v.downsample_bg, &v.view, &label, writes);
            passes += 1;
        }

        let present_bg = self
            .present_bind_group
            .as_ref()
            .expect("present_bind_group not set — graph.resize must run before render");
        let present_writes = self
            .perf
            .as_mut()
            .and_then(|p| p.next_writes("present-blit"));
        self.present_blit.record(
            ctx.encoder,
            present_bg,
            ctx.output,
            "present-blit:pass",
            present_writes,
        );
        passes += 1;

        // PiP overlays — draw a solid border behind each viewer thumbnail
        // and then blit the thumbnail on top of it. Skip silently when a
        // thumbnail would overflow the canvas in either axis so a
        // misconfigured offset doesn't blow up the render.
        for i in 0..self.viewers.len() {
            let Some(v) = &self.viewers[i] else { continue };
            let [x, y] = self.viewer_configs[i].position;
            if x + v.width > ctx.width || y + v.height > ctx.height {
                continue;
            }

            // Border pass: expand the rect by VIEWER_BORDER_PX on every
            // side, clamped to the canvas. Drawing the solid fill first
            // and then the thumbnail on top leaves a visible ring.
            let bx = x.saturating_sub(VIEWER_BORDER_PX);
            let by = y.saturating_sub(VIEWER_BORDER_PX);
            let bw = (v.width + VIEWER_BORDER_PX * 2).min(ctx.width - bx);
            let bh = (v.height + VIEWER_BORDER_PX * 2).min(ctx.height - by);
            let border_label = format!("{}:viewer:border-pass", self.nodes[i].id);
            let border_writes = self
                .perf
                .as_mut()
                .and_then(|p| p.next_writes(&self.viewer_border_labels[i]));
            self.border_fill.record_overlay(
                ctx.encoder,
                ctx.output,
                [bx as f32, by as f32, bw as f32, bh as f32],
                &border_label,
                border_writes,
            );
            passes += 1;

            // Thumbnail blit on top.
            let label = format!("{}:viewer:pip-pass", self.nodes[i].id);
            let writes = self
                .perf
                .as_mut()
                .and_then(|p| p.next_writes(&self.viewer_pip_labels[i]));
            self.present_blit.record_overlay(
                ctx.encoder,
                &v.pip_bg,
                ctx.output,
                [x as f32, y as f32, v.width as f32, v.height as f32],
                &label,
                writes,
            );
            passes += 1;
        }

        // Resolve query set + copy into a readback buffer (no-op when the
        // perf monitor isn't enabled).
        if let Some(p) = &mut self.perf {
            p.end_frame(ctx.encoder);
        }

        FrameStats { passes }
    }

    /// Snapshot of the graph's structure + recent per-node timings. The
    /// HUD uses this to render the dependency tree, memory column, and
    /// dispatch-time column. Cheap to compute (linear in node count).
    pub fn summary(&self) -> GraphSummary {
        // Depth in topological order: each node's depth is
        // `max(input.depth) + 1`, leaves get 0.
        let mut depths = vec![0u32; self.nodes.len()];
        for &i in &self.schedule {
            let entry = &self.nodes[i];
            if !entry.inputs.is_empty() {
                let max_d = entry.inputs.iter().map(|&j| depths[j]).max().unwrap_or(0);
                depths[i] = max_d + 1;
            }
        }

        let mut total_bytes: u64 = 0;
        let nodes: Vec<NodeSummary> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let mut bytes: u64 = 0;
                if let Some(inter) = &self.intermediates[i] {
                    let s = inter.texture.size();
                    bytes += s.width as u64 * s.height as u64 * 4;
                }
                if let Some(viewer) = &self.viewers[i] {
                    let s = viewer.texture.size();
                    bytes += s.width as u64 * s.height as u64 * 4;
                }
                total_bytes += bytes;
                let dispatch_ms = if entry.dispatch_times.is_empty() {
                    0.0
                } else {
                    let sum: f32 = entry.dispatch_times.iter().sum();
                    sum / entry.dispatch_times.len() as f32 * 1000.0
                };
                let gpu_ms = self
                    .perf
                    .as_ref()
                    .and_then(|p| p.timing_ms(&self.schedule_labels[i]));
                NodeSummary {
                    id: entry.id.clone(),
                    depth: depths[i],
                    bytes,
                    is_present: i == self.present_index,
                    viewer_enabled: self.viewer_configs[i].enabled,
                    dispatch_ms,
                    gpu_ms,
                }
            })
            .collect();

        GraphSummary { nodes, total_bytes }
    }

    /// Sum of bytes the graph currently has allocated in intermediate
    /// and viewer textures (assumes 4 bytes/pixel — Rgba8). Powers the
    /// `Mem:` line of the HUD.
    pub fn allocated_bytes(&self) -> u64 {
        let mut bytes: u64 = 0;
        for inter in &self.intermediates {
            if let Some(i) = inter {
                let s = i.texture.size();
                bytes += s.width as u64 * s.height as u64 * 4;
            }
        }
        for viewer in &self.viewers {
            if let Some(v) = viewer {
                let s = v.texture.size();
                bytes += s.width as u64 * s.height as u64 * 4;
            }
        }
        bytes
    }

    /// Returns the on-screen rectangle of a viewer thumbnail in physical
    /// pixels (`[x, y, width, height]`), or `None` if the index is out
    /// of range or that node has no viewer enabled.
    pub fn viewer_rect(&self, viewer_index: usize) -> Option<[u32; 4]> {
        let v = self.viewers.get(viewer_index)?.as_ref()?;
        let cfg = self.viewer_configs.get(viewer_index)?;
        Some([cfg.position[0], cfg.position[1], v.width, v.height])
    }

    /// Update a viewer's pixel offset at runtime (e.g. from a drag
    /// interaction). Returns `false` if the index is out of range.
    pub fn set_viewer_position(&mut self, viewer_index: usize, position: [u32; 2]) -> bool {
        match self.viewer_configs.get_mut(viewer_index) {
            Some(cfg) => {
                cfg.position = position;
                true
            }
            None => false,
        }
    }

    /// Returns the index of the topmost viewer whose rect contains
    /// `cursor` (physical pixels), or `None`. Iterates back-to-front so
    /// later-rendered (visually topmost) thumbnails win when overlapping.
    pub fn hit_test_viewer(&self, cursor: [u32; 2]) -> Option<usize> {
        for i in (0..self.viewers.len()).rev() {
            let Some([x, y, w, h]) = self.viewer_rect(i) else {
                continue;
            };
            if cursor[0] >= x && cursor[0] < x + w && cursor[1] >= y && cursor[1] < y + h {
                return Some(i);
            }
        }
        None
    }

    /// Returns a slot for every node where `viewer.enabled = true`. The
    /// `view` references a texture filled by the most recent `render()`
    /// call; sample it from a downstream UI pipeline to draw the
    /// thumbnail.
    pub fn viewer_textures(&self) -> Vec<ViewerSlot<'_>> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, entry)| {
                self.viewers[i].as_ref().map(|v| ViewerSlot {
                    node_id: entry.id.as_str(),
                    view: &v.view,
                    width: v.width,
                    height: v.height,
                })
            })
            .collect()
    }
}

/// DFS post-order topological sort. Returns indices in execution order
/// (every input precedes its consumer). Reports the first node it finds
/// inside a cycle.
fn topo_sort(
    nodes: &[NodeConfig],
    inputs: &[Vec<usize>],
) -> Result<Vec<usize>, GraphError> {
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum Mark {
        Unseen,
        InProgress,
        Done,
    }

    let n = nodes.len();
    let mut marks = vec![Mark::Unseen; n];
    let mut order = Vec::with_capacity(n);

    // Iterative DFS using an explicit stack so we don't blow Rust's stack
    // on deep graphs. Each stack entry is (node_idx, next_input_to_visit).
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut on_stack: HashSet<usize> = HashSet::new();

    for start in 0..n {
        if marks[start] != Mark::Unseen {
            continue;
        }
        stack.push((start, 0));
        on_stack.insert(start);
        marks[start] = Mark::InProgress;

        while let Some(&(top, next)) = stack.last() {
            if next < inputs[top].len() {
                let upstream = inputs[top][next];
                // Advance the cursor on the parent.
                stack.last_mut().unwrap().1 += 1;
                match marks[upstream] {
                    Mark::Done => {}
                    Mark::InProgress => {
                        return Err(GraphError::CycleInvolving {
                            node: nodes[upstream].id.clone(),
                        });
                    }
                    Mark::Unseen => {
                        marks[upstream] = Mark::InProgress;
                        on_stack.insert(upstream);
                        stack.push((upstream, 0));
                    }
                }
            } else {
                stack.pop();
                on_stack.remove(&top);
                marks[top] = Mark::Done;
                order.push(top);
            }
        }
    }

    Ok(order)
}

// ---------------------------------------------------------------------------
// PerfMonitor — wgpu timestamp queries with non-blocking readback.
//
// One QuerySet sized 2 × `PERF_PASS_CAPACITY` (begin + end timestamps per
// pass). Each frame the graph hands out `RenderPassTimestampWrites` from
// `next_writes(label)`; at end_frame we resolve the query set into an
// intermediate buffer and copy that into one of two readback buffers.
// `after_submit` ingests any prior frame whose map_async has completed and
// kicks off mapping for the current frame on the OTHER readback buffer.
// Skipping a frame happens silently when both buffers are still in-flight.
// ---------------------------------------------------------------------------

const PERF_PASS_CAPACITY: u32 = 32;
const PERF_QUERY_CAPACITY: u32 = PERF_PASS_CAPACITY * 2;
const PERF_BUFFER_BYTES: u64 = (PERF_QUERY_CAPACITY as u64) * 8;
/// Smoothing window — same as lib::FRAME_WINDOW.
const PERF_AVG_WINDOW: usize = 60;

pub struct PerfMonitor {
    query_set: wgpu::QuerySet,
    period_ns_per_tick: f32,
    /// `QUERY_RESOLVE | COPY_SRC` intermediate; `resolve_query_set` writes here.
    resolve_buffer: wgpu::Buffer,
    /// Two CPU-readable buffers we ping-pong between to keep readback
    /// non-blocking. At most one is being written this frame; at most one
    /// is mid-`map_async` from a prior frame.
    readbacks: [PerfReadback; 2],
    /// Per-frame allocation cursor (how many query indices have been
    /// handed out by `next_writes` this frame).
    next_query: u32,
    /// Labels for the queries we've issued this frame, paralleling
    /// `next_query/2` entries. Moves into the readback when we resolve.
    pending_labels: Vec<String>,
    /// Smoothed per-label cook times in ms. Updated as buffers map back.
    timings: std::collections::HashMap<String, std::collections::VecDeque<f32>>,
    /// True when end_frame has actually copied data into a readback this
    /// frame and we should kick off its map_async after submit.
    pending_write_idx: Option<usize>,
}

struct PerfReadback {
    buffer: wgpu::Buffer,
    /// Labels for the frame whose data is parked in this buffer (or
    /// being mapped). `len` ≤ PERF_PASS_CAPACITY.
    labels: Vec<String>,
    /// Number of u64 query slots actually populated for the parked frame.
    query_count: u32,
    /// `map_async` outstanding on this buffer.
    in_flight: bool,
    /// Receiver for the `map_async` completion. None if not in flight.
    rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl PerfMonitor {
    pub fn new(device: &wgpu::Device, period_ns_per_tick: f32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("perf:query-set"),
            ty: wgpu::QueryType::Timestamp,
            count: PERF_QUERY_CAPACITY,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perf:resolve"),
            size: PERF_BUFFER_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let make_readback = |i: usize| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("perf:readback[{i}]")),
                size: PERF_BUFFER_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            PerfReadback {
                buffer,
                labels: Vec::with_capacity(PERF_PASS_CAPACITY as usize),
                query_count: 0,
                in_flight: false,
                rx: None,
            }
        };
        Self {
            query_set,
            period_ns_per_tick,
            resolve_buffer,
            readbacks: [make_readback(0), make_readback(1)],
            next_query: 0,
            pending_labels: Vec::with_capacity(PERF_PASS_CAPACITY as usize),
            timings: std::collections::HashMap::new(),
            pending_write_idx: None,
        }
    }

    pub fn begin_frame(&mut self) {
        self.next_query = 0;
        self.pending_labels.clear();
        self.pending_write_idx = None;
    }

    /// Reserve two timestamp slots and return the descriptor to plug into
    /// the next `begin_render_pass`. Returns `None` once we exhaust the
    /// per-frame budget (`PERF_PASS_CAPACITY`).
    pub fn next_writes(&mut self, label: &str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        if self.next_query + 2 > PERF_QUERY_CAPACITY {
            return None;
        }
        let begin = self.next_query;
        self.next_query += 2;
        self.pending_labels.push(label.to_string());
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(begin + 1),
        })
    }

    /// Insert resolve + copy commands into `encoder`. Skips the frame if
    /// both readbacks are still mid-flight (data dropped silently — the
    /// HUD just shows the most recent successful sample). Picks an idle
    /// readback buffer; remembers it for `after_submit` to map.
    pub fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.next_query == 0 {
            return;
        }
        let target = if !self.readbacks[0].in_flight {
            0
        } else if !self.readbacks[1].in_flight {
            1
        } else {
            // Both buffers are still mapping from earlier frames; skip
            // resolving this frame. Smoothed timings just keep their
            // most-recent values.
            return;
        };

        encoder.resolve_query_set(
            &self.query_set,
            0..self.next_query,
            &self.resolve_buffer,
            0,
        );
        let bytes = (self.next_query as u64) * 8;
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readbacks[target].buffer,
            0,
            bytes,
        );
        let rb = &mut self.readbacks[target];
        rb.labels.clear();
        rb.labels.extend(self.pending_labels.iter().cloned());
        rb.query_count = self.next_query;
        self.pending_write_idx = Some(target);
    }

    /// After `queue.submit`: ingest any completed prior-frame readbacks,
    /// then start `map_async` on this frame's readback if `end_frame`
    /// chose one. Drives the wgpu callback queue via `device.poll(Poll)`
    /// (no-op on web; native needs it to fire callbacks).
    pub fn after_submit(&mut self, device: &wgpu::Device) {
        // Ingest completed prior frames.
        for i in 0..2 {
            let rx_done = match &self.readbacks[i].rx {
                Some(rx) => matches!(rx.try_recv(), Ok(Ok(()))),
                None => false,
            };
            if rx_done {
                self.ingest(i);
            }
        }

        // Kick off map_async on this frame's readback.
        if let Some(i) = self.pending_write_idx.take() {
            let bytes = (self.readbacks[i].query_count as u64) * 8;
            let slice = self.readbacks[i].buffer.slice(0..bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.readbacks[i].in_flight = true;
            self.readbacks[i].rx = Some(rx);
        }

        // Drive callbacks; no-op on web (JS event loop handles it).
        let _ = device.poll(wgpu::Maintain::Poll);
    }

    fn ingest(&mut self, i: usize) {
        let labels = std::mem::take(&mut self.readbacks[i].labels);
        let bytes = (self.readbacks[i].query_count as u64) * 8;
        {
            let slice = self.readbacks[i].buffer.slice(0..bytes);
            let data = slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);
            for (idx, label) in labels.iter().enumerate() {
                let begin = timestamps[idx * 2];
                let end = timestamps[idx * 2 + 1];
                let ticks = end.saturating_sub(begin);
                let ms = (ticks as f32 * self.period_ns_per_tick) / 1_000_000.0;
                let entry = self
                    .timings
                    .entry(label.clone())
                    .or_insert_with(|| {
                        std::collections::VecDeque::with_capacity(PERF_AVG_WINDOW)
                    });
                if entry.len() == PERF_AVG_WINDOW {
                    entry.pop_front();
                }
                entry.push_back(ms);
            }
        }
        self.readbacks[i].buffer.unmap();
        self.readbacks[i].in_flight = false;
        self.readbacks[i].rx = None;
        self.readbacks[i].query_count = 0;
    }

    /// Most recent rolling-average ms for `label`, or `None` if no samples
    /// have come back yet.
    pub fn timing_ms(&self, label: &str) -> Option<f32> {
        self.timings.get(label).filter(|d| !d.is_empty()).map(|d| {
            let sum: f32 = d.iter().sum();
            sum / d.len() as f32
        })
    }
}
