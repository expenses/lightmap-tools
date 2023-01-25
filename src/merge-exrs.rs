use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();

    let base = parking_lot::Mutex::new(load_image(&args[0]));

    // Note: exr decompression is already parallelised and this only slightly improves
    // performance.
    args[1..].par_iter().for_each(|path| {
        let (new_image, new_samples) = load_image(&path);

        let mut base = base.lock();

        base.1 += new_samples;

        for (base, new) in (&mut *base.0).iter_mut().zip(&*new_image) {
            *base += *new;
        }
    });

    let (mut base, total_samples) = base.into_inner();

    for pixel in base.pixels_mut() {
        for subpixel in &mut pixel.0 {
            *subpixel /= total_samples as f32;
        }
    }

    base.save("out.exr").unwrap();

    dbg!(&total_samples);
}

fn load_image(path: &str) -> (image::ImageBuffer<image::Rgb<f32>, Vec<f32>>, u32) {
    let (_, second_chunk) = path.rsplit_once("_").unwrap();
    let (num_samples, _) = second_chunk.split_once(".").unwrap();
    let num_samples: u32 = num_samples.parse().unwrap();

    let mut image = image::open(path).unwrap().to_rgb32f();

    for subpixel in &mut *image {
        *subpixel *= num_samples as f32;

        if subpixel.is_nan() {
            *subpixel = 0.0;
        }
    }

    (image, num_samples)
}
