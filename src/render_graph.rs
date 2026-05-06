//! Minimal render graph — a list of nodes loaded from `config/graph.toml`.
//!
//! Designed roughly after bevy 0.18's `bevy_render::render_graph` (Node trait,
//! slot/edge model, topological execute) but stripped of bevy-isms (no ECS,
//! no sub-graphs, no view-driven dispatch). The whole module fits in one
//! file because the v1 graph supports only a single fragment-shader node
//! with no inputs.
//!
//! Adding a second node type means: implement `Node`, register it in
//! `RenderGraph::from_config`'s match on `NodeConfig::kind`, and the graph
//! walks them in declared order. The chaining/intermediate-texture path
//! lights up once a non-leaf node is added.

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
    /// nodes it'll be an intermediate texture managed by the graph.
    pub output: &'a wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub time: f32,
    /// Cursor position in physical pixels (winit y-down).
    pub mouse: [f32; 2],
}

/// A render-graph node. Implementations own their pipeline + per-node
/// uniform buffer + bind group; `record` issues whatever GPU work is needed
/// to fill `ctx.output`.
///
/// No `Send + Sync` bound — wgpu's pipeline/buffer/bind-group handles are
/// `!Send + !Sync` on wasm32 (they wrap JS handles), and the graph is only
/// ever walked from the render thread.
pub trait Node {
    fn record(&self, ctx: &mut NodeContext<'_>);
}

/// Generic fragment-shader node. Constructor takes raw GLSL source for both
/// the vertex and fragment stages so any pair of shaders that follow the
/// standard layout (no vertex inputs, fragment binds `Uniforms` at
/// `set=0, binding=0`) can be loaded as a node. Reads no input textures —
/// the chaining path with input slots is the next slice of work.
///
/// `label` is a short identifier (the node id from the TOML config) used
/// for wgpu debug labels — `<label>:pipeline`, `<label>:pass`, etc.
pub struct GlslNode {
    label: String,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl GlslNode {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        vert_src: &str,
        frag_src: &str,
        label: &str,
    ) -> Self {
        // GLSL → naga → backend SPIR-V/MSL/WGSL/GLSL ES.
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

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}:bind-layout")),
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
            label: Some(&format!("{label}:bind-group")),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
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
            bind_group,
        }
    }
}

impl Node for GlslNode {
    fn record(&self, ctx: &mut NodeContext<'_>) {
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
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

// ---------------------------------------------------------------------------
// Compile-time shader registry
// ---------------------------------------------------------------------------

/// Look up a shader by short name. The wasm bundle can't read the
/// filesystem so every shader pair must be embedded at compile time via
/// `include_str!`. Add new entries here as new shaders are added.
///
/// Returns `(vert_src, frag_src)`.
pub fn embedded_shader(name: &str) -> Option<(&'static str, &'static str)> {
    match name {
        "mandelbrot" => Some((
            include_str!("../shaders/mandelbrot.vert"),
            include_str!("../shaders/mandelbrot.frag"),
        )),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Config (TOML)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct GraphConfig {
    #[serde(rename = "node", default)]
    pub nodes: Vec<NodeConfig>,
}

#[derive(Debug, Deserialize)]
pub struct NodeConfig {
    pub id: String,
    pub kind: String,
    /// Required when `kind = "glsl"` — the short name of an entry in the
    /// `embedded_shader` registry (e.g. `"mandelbrot"`).
    #[serde(default)]
    pub shader: Option<String>,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub present: bool,
}

/// Parse a TOML graph definition. Returns a structured error on bad input
/// rather than panicking.
pub fn parse_config(toml_src: &str) -> Result<GraphConfig, String> {
    toml::from_str(toml_src).map_err(|e| format!("graph.toml parse error: {e}"))
}

/// Compile-time embedded default graph definition.
pub const DEFAULT_GRAPH_TOML: &str = include_str!("../config/graph.toml");

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum GraphError {
    Config(String),
    UnknownNodeKind { id: String, kind: String },
    /// `kind = "glsl"` requires a `shader` field naming an `embedded_shader`
    /// registry entry.
    MissingShaderField { id: String },
    /// `shader = "<name>"` didn't match any entry in `embedded_shader`.
    UnknownShader { id: String, shader: String },
    MissingPresentNode,
    MultiplePresentNodes,
    /// v1 only supports nodes with empty `inputs`. Removed once the chaining
    /// path is wired up.
    InputsNotYetSupported { id: String },
    /// The named upstream id wasn't declared in the config.
    UnknownInput { node: String, input: String },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::Config(s) => write!(f, "{s}"),
            GraphError::UnknownNodeKind { id, kind } => {
                write!(f, "node {id:?} has unknown kind {kind:?}")
            }
            GraphError::MissingShaderField { id } => {
                write!(f, "node {id:?} (kind = \"glsl\") is missing the `shader` field")
            }
            GraphError::UnknownShader { id, shader } => {
                write!(
                    f,
                    "node {id:?} references shader {shader:?} which is not in the embedded_shader registry"
                )
            }
            GraphError::MissingPresentNode => write!(f, "no node has `present = true`"),
            GraphError::MultiplePresentNodes => write!(f, "more than one node has `present = true`"),
            GraphError::InputsNotYetSupported { id } => {
                write!(f, "node {id:?} declares inputs; chaining is not implemented yet")
            }
            GraphError::UnknownInput { node, input } => {
                write!(f, "node {node:?} references unknown input {input:?}")
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// A built render graph: nodes + the order in which to walk them.
pub struct RenderGraph {
    /// In execution order. v1 schedule == config order since there's only
    /// one node; once chaining lands this becomes the topological sort.
    nodes: Vec<Box<dyn Node>>,
}

impl RenderGraph {
    /// Build the graph from a parsed config. `present_format` is the wgpu
    /// texture format the present node will write into — typically the
    /// surface's view format (sRGB sibling on web, sRGB native on Mac).
    pub fn build(
        device: &wgpu::Device,
        present_format: wgpu::TextureFormat,
        cfg: &GraphConfig,
    ) -> Result<Self, GraphError> {
        // Validate `present` count.
        let present_ids: Vec<&str> = cfg
            .nodes
            .iter()
            .filter(|n| n.present)
            .map(|n| n.id.as_str())
            .collect();
        match present_ids.len() {
            0 => return Err(GraphError::MissingPresentNode),
            1 => {}
            _ => return Err(GraphError::MultiplePresentNodes),
        }

        // Validate inputs reference real nodes — even if v1 rejects them
        // outright, this keeps the error message useful.
        let known: std::collections::HashSet<&str> =
            cfg.nodes.iter().map(|n| n.id.as_str()).collect();
        for n in &cfg.nodes {
            for input in &n.inputs {
                if !known.contains(input.as_str()) {
                    return Err(GraphError::UnknownInput {
                        node: n.id.clone(),
                        input: input.clone(),
                    });
                }
            }
            if !n.inputs.is_empty() {
                return Err(GraphError::InputsNotYetSupported { id: n.id.clone() });
            }
        }

        // Instantiate nodes.
        let mut nodes: Vec<Box<dyn Node>> = Vec::with_capacity(cfg.nodes.len());
        for n in &cfg.nodes {
            // For v1 every node renders to the present format. Once we add
            // intermediate textures, non-present nodes will pick their own.
            let format = present_format;
            let node: Box<dyn Node> = match n.kind.as_str() {
                "glsl" => {
                    let shader = n
                        .shader
                        .as_deref()
                        .ok_or_else(|| GraphError::MissingShaderField { id: n.id.clone() })?;
                    let (vert_src, frag_src) =
                        embedded_shader(shader).ok_or_else(|| GraphError::UnknownShader {
                            id: n.id.clone(),
                            shader: shader.to_string(),
                        })?;
                    Box::new(GlslNode::new(device, format, vert_src, frag_src, &n.id))
                }
                other => {
                    return Err(GraphError::UnknownNodeKind {
                        id: n.id.clone(),
                        kind: other.to_string(),
                    })
                }
            };
            nodes.push(node);
        }

        log::info!("render graph built: {} node(s)", nodes.len());
        Ok(Self { nodes })
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

    /// Execute every node in schedule order, recording into `ctx.encoder`.
    /// The caller submits the encoder.
    pub fn render(&self, ctx: &mut NodeContext<'_>) {
        for node in &self.nodes {
            node.record(ctx);
        }
    }
}
