fn main() {
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");

    cc::Build::new()
        .file("src/metal_bridge.m")
        .flag("-x")
        .flag("objective-c")
        .flag("-fobjc-arc")
        .compile("metal_bridge");
}
