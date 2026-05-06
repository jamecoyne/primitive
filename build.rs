//! Build script — scans `shaders/` for matching `<name>.vert` + `<name>.frag`
//! pairs and emits a `shader_registry.rs` file with an `embedded_shader`
//! function that resolves a name to the two source strings via
//! `include_str!`. The generated file is `include!`-ed from
//! `src/render_graph.rs`.
//!
//! Adding a new shader: drop `shaders/<name>.vert` and `shaders/<name>.frag`,
//! then reference `shader = "<name>"` in `config/graph.toml`. No Rust edits.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("shaders");

    // Re-run if any shader file is added, removed, or modified.
    println!("cargo:rerun-if-changed={}", shaders_dir.display());

    // Group `.vert`/`.frag` files by stem.
    let mut stems: BTreeMap<String, ShaderPair> = BTreeMap::new();
    for entry in fs::read_dir(&shaders_dir).expect("shaders/ directory missing") {
        let entry = entry.unwrap();
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue };
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        let pair = stems.entry(stem.to_string()).or_default();
        match ext {
            "vert" => pair.vert = Some(path.clone()),
            "frag" => pair.frag = Some(path.clone()),
            _ => {}
        }
        // Per-file rerun is implied by the directory rerun above, but listing
        // each one keeps cargo's incremental graph happy when the directory
        // mtime doesn't change.
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let mut out = String::new();
    out.push_str("// Auto-generated from shaders/ by build.rs. Do not edit.\n");
    out.push_str("//\n");
    out.push_str("// Add a shader by dropping <name>.vert and <name>.frag under\n");
    out.push_str("// shaders/, then reference it from config/graph.toml as\n");
    out.push_str("// `shader = \"<name>\"`. No Rust edit needed.\n\n");
    out.push_str("pub fn embedded_shader(name: &str) -> Option<(&'static str, &'static str)> {\n");
    out.push_str("    match name {\n");

    let mut emitted = 0;
    for (stem, pair) in &stems {
        let (Some(vert), Some(frag)) = (&pair.vert, &pair.frag) else {
            // A bare .vert or .frag without its mate is ignored — skip with a
            // build-time warning so the user knows nothing was registered.
            println!(
                "cargo:warning=shaders/{stem}.* is missing its {} mate; skipping",
                if pair.vert.is_some() { ".frag" } else { ".vert" },
            );
            continue;
        };
        // include_str! requires a literal path string. Forward slashes work
        // on every host wgpu supports.
        let vert_lit = vert.display().to_string().replace('\\', "/");
        let frag_lit = frag.display().to_string().replace('\\', "/");
        out.push_str(&format!(
            "        {stem:?} => Some((include_str!({vert_lit:?}), include_str!({frag_lit:?}))),\n"
        ));
        emitted += 1;
    }
    out.push_str("        _ => None,\n");
    out.push_str("    }\n");
    out.push_str("}\n");

    if emitted == 0 {
        panic!("build.rs: no shader pairs found under shaders/");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_path = out_dir.join("shader_registry.rs");
    fs::write(&out_path, out).unwrap();
}

#[derive(Default)]
struct ShaderPair {
    vert: Option<PathBuf>,
    frag: Option<PathBuf>,
}
