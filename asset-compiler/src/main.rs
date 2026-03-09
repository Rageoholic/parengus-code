mod mesh;

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = run(&args);
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    match args.get(1).map(String::as_str) {
        Some("mesh") => {
            if args.len() < 4 {
                return Err("usage: asset-compiler mesh \
                     <input.gltf> <output.pmesh> \
                     [--tex-ref <name>]"
                    .to_string());
            }
            let src = Path::new(&args[2]);
            let dst = Path::new(&args[3]);
            let tex_ref = args
                .windows(2)
                .find(|w| w[0] == "--tex-ref")
                .map(|w| w[1].as_str());
            mesh::compile(src, dst, tex_ref)
        }
        _ => Err("usage: asset-compiler <mesh|image> ...".to_string()),
    }
}
