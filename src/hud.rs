//! Performance HUD — alpha-blended overlay drawn on top of the swapchain
//! after the render graph finishes. Hosts a small CPU bitmap that the
//! caller updates with `set_text`, uploads to a GPU texture, then draws
//! as a textured quad clipped to a viewport rect.
//!
//! The font is the 8×8 ASCII bitmap from the `font8x8` crate. White text
//! on a translucent black background — readable on any swapchain content.

use font8x8::legacy::BASIC_LEGACY;

/// HUD bitmap dimensions. 8×8 font with no inter-character spacing fits
/// roughly 36 chars across at this width. Height fits a header line, up to
/// ~6 lines of dependency tree, and 3 lines of totals — about 10 lines.
pub const HUD_WIDTH: u32 = 296;
pub const HUD_HEIGHT: u32 = 96;

/// Translucent black background pixel, premultiplied for the alpha blend.
const BG_RGBA: [u8; 4] = [0, 0, 0, 180];
const FG_RGBA: [u8; 4] = [255, 255, 255, 255];

const HUD_VERT: &str = "#version 450
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

const HUD_FRAG: &str = "#version 450
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;
layout(set = 0, binding = 0) uniform sampler   s_in;
layout(set = 0, binding = 1) uniform texture2D t_in;
void main() {
    // Texture y=0 is the top of the bitmap; v_uv y=0 is the bottom of
    // the viewport. Flip y so the rasterized text reads upright.
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    o_color = texture(sampler2D(t_in, s_in), uv);
}
";

pub struct Hud {
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    sampler: wgpu::Sampler,
    #[allow(dead_code)]
    bind_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    texture: wgpu::Texture,
    pub width: u32,
    pub height: u32,
    /// CPU-side RGBA bitmap. Re-rasterized on every `set_text` call.
    pixels: Vec<u8>,
}

impl Hud {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let vs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud.vert"),
            source: wgpu::ShaderSource::Glsl {
                shader: HUD_VERT.into(),
                stage: wgpu::naga::ShaderStage::Vertex,
                defines: Default::default(),
            },
        });
        let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud.frag"),
            source: wgpu::ShaderSource::Glsl {
                shader: HUD_FRAG.into(),
                stage: wgpu::naga::ShaderStage::Fragment,
                defines: Default::default(),
            },
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hud:sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hud:bind-layout"),
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
            label: Some("hud:pipeline-layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud:pipeline"),
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
                    // Standard alpha blending so the translucent-black
                    // background blends with whatever the graph drew.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hud:texture"),
            size: wgpu::Extent3d {
                width: HUD_WIDTH,
                height: HUD_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Linear (no sRGB) — the bitmap holds final display values, no
            // gamma decode needed when sampling.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud:bind-group"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });

        let pixels = vec![0u8; (HUD_WIDTH * HUD_HEIGHT * 4) as usize];

        Self {
            pipeline,
            sampler,
            bind_layout,
            bind_group,
            texture,
            width: HUD_WIDTH,
            height: HUD_HEIGHT,
            pixels,
        }
    }

    /// Re-rasterize the HUD bitmap from `text` and upload it. Newlines
    /// move to the next row (8 px + 1 px line spacing).
    pub fn set_text(&mut self, queue: &wgpu::Queue, text: &str) {
        // Fill background.
        for chunk in self.pixels.chunks_exact_mut(4) {
            chunk[0] = BG_RGBA[0];
            chunk[1] = BG_RGBA[1];
            chunk[2] = BG_RGBA[2];
            chunk[3] = BG_RGBA[3];
        }

        const PAD_X: u32 = 6;
        const PAD_Y: u32 = 4;
        const LINE_HEIGHT: u32 = 9;

        let mut x = PAD_X;
        let mut y = PAD_Y;
        for ch in text.chars() {
            if ch == '\n' {
                y += LINE_HEIGHT;
                x = PAD_X;
                continue;
            }
            // Non-printable / out-of-range → render as a space.
            let glyph = if (ch as usize) < BASIC_LEGACY.len() {
                BASIC_LEGACY[ch as usize]
            } else {
                BASIC_LEGACY[b' ' as usize]
            };
            for row in 0..8u32 {
                let bits = glyph[row as usize];
                for col in 0..8u32 {
                    if bits & (1 << col) != 0 {
                        let px = x + col;
                        let py = y + row;
                        if px < self.width && py < self.height {
                            let idx = ((py * self.width + px) * 4) as usize;
                            self.pixels[idx]     = FG_RGBA[0];
                            self.pixels[idx + 1] = FG_RGBA[1];
                            self.pixels[idx + 2] = FG_RGBA[2];
                            self.pixels[idx + 3] = FG_RGBA[3];
                        }
                    }
                }
            }
            x += 8;
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Draw the HUD bitmap into a sub-rect of `target` with alpha blending.
    /// `viewport` is `[x, y, width, height]` in physical pixels (top-left
    /// origin).
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        viewport: [f32; 4],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hud:pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Preserve everything the graph already drew.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_viewport(viewport[0], viewport[1], viewport[2], viewport[3], 0.0, 1.0);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Format `wgpu::Backend` as a short label for the HUD line. Maps the
/// browser variants to "WebGPU" / "GLES" so the HUD reads naturally on
/// either platform.
pub fn backend_label(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::BrowserWebGpu => "WebGPU",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Dx12 => "D3D12",
        wgpu::Backend::Gl => "GLES",
        wgpu::Backend::Empty => "(none)",
    }
}

/// Format the GPU device name for the HUD. wgpu reports an empty `name`
/// on the browser, so we fall back to the backend in that case.
pub fn gpu_label(info: &wgpu::AdapterInfo) -> String {
    if info.name.is_empty() {
        format!("(browser {})", backend_label(info.backend))
    } else {
        info.name.clone()
    }
}
