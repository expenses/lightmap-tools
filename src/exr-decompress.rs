fn main() {
    let filename = std::env::args().nth(1).unwrap();

    let image = image::open(&filename).unwrap();

    for (i, output) in std::env::args().skip(2).enumerate() {
        let image = image.crop_imm(
            i as u32 * image.width() / 4,
            0,
            image.width() / 4,
            image.height(),
        );
        let image = image.to_rgba32f();

        let bytes: &[u8] = bytemuck::cast_slice(&*image);

        std::fs::write(&output, bytes).unwrap();
    }
}
