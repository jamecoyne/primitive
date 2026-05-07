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
/// No `Send + Sync` bound — wgpu's pipeline/buffer/bind-group handles are
/// `!Send + !Sync` on wasm32 (they wrap JS handles), and the graph is only
/// ever walked from the render thread.
pub trait Node {
    fn record(&self, ctx: &mut NodeContext<'_>);
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
    fn record(&self, ctx: &mut NodeContext<'_>) {
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
            timestamp_writes: None,
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
/// "viewer active" toggle and viewer resolution: when `enabled`, the node's
/// full intermediate is downsampled into a thumbnail-sized texture every
/// frame and overlaid in the named corner of the swapchain (PiP).
///
/// TOML form (inline table per node):
///
/// ```toml
/// viewer = { enabled = true, resolution = [128, 96], position = "top-right" }
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
    /// Which corner of the swapchain the PiP thumbnail anchors to.
    /// Multiple viewers in the same corner stack: top corners stack
    /// downward, bottom corners stack upward.
    #[serde(default)]
    pub position: ViewerPosition,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            resolution: default_viewer_resolution(),
            position: ViewerPosition::default(),
        }
    }
}

fn default_viewer_resolution() -> [u32; 2] {
    [128, 96]
}

/// Corner anchor for a viewer's PiP overlay. Strings in TOML are
/// kebab-case (`"top-left"`, `"top-right"`, `"bottom-left"`,
/// `"bottom-right"`).
#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ViewerPosition {
    #[default]
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
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
            timestamp_writes: None,
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
            timestamp_writes: None,
        });
        pass.set_viewport(
            viewport[0], viewport[1], viewport[2], viewport[3], 0.0, 1.0,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
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
}

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
    width: u32,
    height: u32,
}

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
            });
        }

        let intermediates = (0..nodes.len()).map(|_| None).collect();
        let viewers = (0..nodes.len()).map(|_| None).collect();
        let viewer_configs: Vec<ViewerConfig> =
            cfg.nodes.iter().map(|n| n.viewer.clone()).collect();
        let present_blit = Blitter::new(device, present_format, "present-blit");
        let viewer_blit = Blitter::new(device, INTERMEDIATE_FORMAT, "viewer-blit");

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
            width: 0,
            height: 0,
        })
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
    /// into `ctx.output`.
    pub fn render(&self, ctx: &mut NodeContext<'_>) {
        for &node_idx in &self.schedule {
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
            self.nodes[node_idx].impl_.record(&mut sub_ctx);
        }

        // Viewer downsamples — independent of present, can run in any
        // order relative to it. Must precede the PiP overlays below since
        // those sample the textures these passes write.
        for (i, viewer) in self.viewers.iter().enumerate() {
            if let Some(v) = viewer {
                let label = format!("{}:viewer:downsample-pass", self.nodes[i].id);
                self.viewer_blit
                    .record(ctx.encoder, &v.downsample_bg, &v.view, &label);
            }
        }

        let present_bg = self
            .present_bind_group
            .as_ref()
            .expect("present_bind_group not set — graph.resize must run before render");
        self.present_blit
            .record(ctx.encoder, present_bg, ctx.output, "present-blit:pass");

        // PiP overlays — draw each viewer thumbnail at its configured
        // corner. Multiple viewers in the same corner stack: top corners
        // stack down, bottom corners stack up. Skip silently when a
        // thumbnail wouldn't fit so a tiny viewport doesn't blow up.
        const MARGIN: u32 = 16;
        let mut cursors = CornerCursors::new(ctx.width, ctx.height, MARGIN);
        for (i, viewer) in self.viewers.iter().enumerate() {
            let Some(v) = viewer else { continue };
            let position = self.viewer_configs[i].position;
            let Some((x, y)) = cursors.place(v.width, v.height, position) else {
                continue;
            };
            let label = format!("{}:viewer:pip-pass", self.nodes[i].id);
            self.present_blit.record_overlay(
                ctx.encoder,
                &v.pip_bg,
                ctx.output,
                [x as f32, y as f32, v.width as f32, v.height as f32],
                &label,
            );
        }
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

/// Stacking layout for PiP viewer overlays. Each call to `place` returns
/// the (x, y) top-left of the next thumbnail at the requested corner, or
/// `None` when there isn't enough room left.
struct CornerCursors {
    margin: u32,
    canvas_w: u32,
    canvas_h: u32,
    /// Top-of-next-thumbnail cursor for the two top corners. Starts at
    /// `margin` and grows downward as thumbnails are placed.
    tl_top: u32,
    tr_top: u32,
    /// Bottom-of-next-thumbnail cursor for the two bottom corners. Starts
    /// at `canvas_h - margin` and shrinks upward as thumbnails are placed.
    bl_bottom: u32,
    br_bottom: u32,
}

impl CornerCursors {
    fn new(canvas_w: u32, canvas_h: u32, margin: u32) -> Self {
        let bottom = canvas_h.saturating_sub(margin);
        Self {
            margin,
            canvas_w,
            canvas_h,
            tl_top: margin,
            tr_top: margin,
            bl_bottom: bottom,
            br_bottom: bottom,
        }
    }

    fn place(&mut self, w: u32, h: u32, position: ViewerPosition) -> Option<(u32, u32)> {
        let m = self.margin;
        // Width bound — thumbnail + two margins must fit horizontally.
        if w + m * 2 > self.canvas_w {
            return None;
        }
        match position {
            ViewerPosition::TopLeft => {
                let y = self.tl_top;
                if y + h + m > self.canvas_h {
                    return None;
                }
                self.tl_top = y + h + m;
                Some((m, y))
            }
            ViewerPosition::TopRight => {
                let y = self.tr_top;
                if y + h + m > self.canvas_h {
                    return None;
                }
                self.tr_top = y + h + m;
                let x = self.canvas_w - m - w;
                Some((x, y))
            }
            ViewerPosition::BottomLeft => {
                if self.bl_bottom < h + m {
                    return None;
                }
                let y = self.bl_bottom - h;
                self.bl_bottom = y.saturating_sub(m);
                Some((m, y))
            }
            ViewerPosition::BottomRight => {
                if self.br_bottom < h + m {
                    return None;
                }
                let y = self.br_bottom - h;
                self.br_bottom = y.saturating_sub(m);
                let x = self.canvas_w - m - w;
                Some((x, y))
            }
        }
    }
}
