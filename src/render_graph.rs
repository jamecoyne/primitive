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
    #[serde(rename = "node", default)]
    pub nodes: Vec<NodeConfig>,
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
// Graph — chaining, intermediates, topological execute
// ---------------------------------------------------------------------------

/// Format used for every intermediate (non-present) texture. Keeping it the
/// same as the present format means cross-platform parity at the swapchain
/// boundary still holds: linear shader output is sRGB-encoded once on
/// every render-pass write.
const INTERMEDIATE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[derive(Debug)]
pub enum GraphError {
    Config(String),
    UnknownNodeKind { id: String, kind: String },
    /// `kind = "glsl"` requires both `vert` and `frag` fields with full
    /// GLSL source. Names which one is missing.
    MissingShaderStage { id: String, stage: &'static str },
    DuplicateNodeId { id: String },
    MissingPresentNode,
    MultiplePresentNodes,
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
            GraphError::MissingPresentNode => write!(f, "no node has `present = true`"),
            GraphError::MultiplePresentNodes => write!(f, "more than one node has `present = true`"),
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
    /// Index in `nodes` of the unique node with `present = true`.
    present_index: usize,
    /// Format the present node renders into (the surface's view format).
    /// Stored for completeness; intermediates use `INTERMEDIATE_FORMAT`.
    #[allow(dead_code)]
    present_format: wgpu::TextureFormat,
    /// One slot per node, in the same order as `nodes`. The slot at
    /// `present_index` stays `None`; every other slot holds the texture
    /// the node renders into and downstream nodes sample from. Reallocated
    /// on `resize`.
    intermediates: Vec<Option<Intermediate>>,
    width: u32,
    height: u32,
}

struct Intermediate {
    #[allow(dead_code)]
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

        // 2. Exactly one present node.
        let present_indices: Vec<usize> = cfg
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.present)
            .map(|(i, _)| i)
            .collect();
        let present_index = match present_indices.as_slice() {
            [] => return Err(GraphError::MissingPresentNode),
            [i] => *i,
            _ => return Err(GraphError::MultiplePresentNodes),
        };

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

        // 5. Instantiate nodes.
        let mut nodes: Vec<NodeEntry> = Vec::with_capacity(cfg.nodes.len());
        for (i, n) in cfg.nodes.iter().enumerate() {
            // Present node renders into the surface format; everything else
            // renders into INTERMEDIATE_FORMAT.
            let format = if i == present_index {
                present_format
            } else {
                INTERMEDIATE_FORMAT
            };
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

        log::info!(
            "render graph built: {} node(s), schedule = {:?}",
            nodes.len(),
            schedule
                .iter()
                .map(|&i| nodes[i].id.as_str())
                .collect::<Vec<_>>()
        );

        Ok(Self {
            nodes,
            schedule,
            present_index,
            present_format,
            intermediates,
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

    /// (Re)allocate intermediate textures and rebuild every node's bind
    /// group. Call this once after `build` and again whenever the canvas
    /// size changes.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.width = width;
        self.height = height;

        // (Re)allocate intermediates for every non-present node.
        for (i, entry) in self.nodes.iter().enumerate() {
            if i == self.present_index {
                self.intermediates[i] = None;
                continue;
            }
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

        // Rebuild every bind group (textures changed; even no-input nodes
        // need their initial bind group built).
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
    }

    /// Walk the schedule, recording each node into `ctx.encoder`. The caller
    /// supplies `ctx.output` for the present node; non-present nodes render
    /// into their owned intermediate views.
    pub fn render(&self, ctx: &mut NodeContext<'_>) {
        for &node_idx in &self.schedule {
            let output: &wgpu::TextureView = if node_idx == self.present_index {
                ctx.output
            } else {
                &self.intermediates[node_idx]
                    .as_ref()
                    .expect("intermediate must be allocated by resize before render")
                    .view
            };
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
