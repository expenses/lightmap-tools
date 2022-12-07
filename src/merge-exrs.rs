fn main() {
    let (mut base, mut total_samples) = load_image(&std::env::args().nth(1).unwrap());

    for path in std::env::args().skip(2) {
        let (new_image, new_samples) = load_image(&path);

        total_samples += new_samples;

        for x in 0..base.width() {
            for y in 0..base.height() {
                let pixel = new_image.get_pixel(x, y);

                let mut dest_pixel = base.get_pixel_mut(x, y);

                dest_pixel.0[0] += pixel.0[0];
                dest_pixel.0[1] += pixel.0[1];
                dest_pixel.0[2] += pixel.0[2];
            }
        }
    }

    for pixel in base.pixels_mut() {
        for subpixel in &mut pixel.0 {
            *subpixel /= total_samples as f32;
        }
    }

    base.save("out.exr").unwrap();

    dbg!(&total_samples);
}

fn load_image(path: &str) -> (image::ImageBuffer<image::Rgb<f32>, Vec<f32>>, u32) {
    let (_, second_chunk) = path.split_once("_").unwrap();
    let (num_samples, _) = second_chunk.split_once(".").unwrap();
    let num_samples: u32 = num_samples.parse().unwrap();

    let mut image = image::open(path).unwrap().to_rgb32f();

    for pixel in image.pixels_mut() {
        for subpixel in &mut pixel.0 {
            *subpixel *= num_samples as f32;
        }
    }

    (image, num_samples)
}
