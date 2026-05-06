//! `cargo run --bin render_frame -- --output frame.png --time 2.5 --mx 0.5 --my 0.5`
//!
//! Renders a single frame using the same wgpu pipeline as the live app and
//! writes it to a PNG. Used by the cross-platform pixel-diff test
//! (tests/cross_platform.mjs) to obtain the native frame.

use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::process::ExitCode;

use wgpuweb::{render_offscreen, RenderConfig};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut output: Option<String> = None;
    let mut time = 2.5_f32;
    let mut mx = 0.5_f32;
    let mut my = 0.5_f32;
    let mut width = 960_u32;
    let mut height = 720_u32;

    let mut i = 0;
    while i < args.len() {
        let next = || {
            args.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("missing value for {}", args[i]);
                std::process::exit(2);
            })
        };
        match args[i].as_str() {
            "--output" | "-o" => {
                output = Some(next());
                i += 2;
            }
            "--time" => {
                time = next().parse().expect("--time must be f32");
                i += 2;
            }
            "--mx" => {
                mx = next().parse().expect("--mx must be f32");
                i += 2;
            }
            "--my" => {
                my = next().parse().expect("--my must be f32");
                i += 2;
            }
            "--width" => {
                width = next().parse().expect("--width must be u32");
                i += 2;
            }
            "--height" => {
                height = next().parse().expect("--height must be u32");
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: render_frame --output <path> [--time <f32>] [--mx <f32>] [--my <f32>] [--width <u32>] [--height <u32>]"
                );
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown arg: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let Some(output) = output else {
        eprintln!("--output is required");
        return ExitCode::from(2);
    };

    let pixels = pollster::block_on(render_offscreen(RenderConfig {
        width,
        height,
        time,
        mouse_uv: [mx, my],
    }));

    let file = File::create(&output).expect("create output png");
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("png header")
        .write_image_data(&pixels)
        .expect("png write");

    eprintln!(
        "wrote {output} ({}x{}, time={time}, mouse_uv=({mx}, {my}))",
        width, height,
    );
    ExitCode::SUCCESS
}
