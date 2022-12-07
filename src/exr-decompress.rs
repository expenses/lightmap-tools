fn main() {
    let filename = std::env::args().nth(1).unwrap();
    let output = std::env::args().nth(2).unwrap();
    let image = image::open(&filename).unwrap();
    let image = image.to_rgba32f();

    let bytes: &[u8] = bytemuck::cast_slice(&*image);

    std::fs::write(&output, bytes).unwrap();
}
