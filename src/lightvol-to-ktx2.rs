use half::f16;
use ktx2_tools::ktx2;
use std::borrow::Cow;

fn main() {
    let filename = std::env::args().nth(1).unwrap();
    let num_z_levels: u32 = std::env::args().nth(2).unwrap().parse().unwrap();

    let base_image = image::open(&filename).unwrap();

    let sh_0 = base_image
        .crop_imm(
            0, //i as u32 * image.width() / 4,
            0,
            base_image.width() / 4,
            base_image.height(),
        )
        .to_rgba32f();

    let mut sh_1_x = base_image
        .crop_imm(
            base_image.width() / 4,
            0,
            base_image.width() / 4,
            base_image.height(),
        )
        .to_rgba32f();

    let mut sh_1_y = base_image
        .crop_imm(
            2 * base_image.width() / 4,
            0,
            base_image.width() / 4,
            base_image.height(),
        )
        .to_rgba32f();

    let mut sh_1_z = base_image
        .crop_imm(
            3 * base_image.width() / 4,
            0,
            base_image.width() / 4,
            base_image.height(),
        )
        .to_rgba32f();

    for (base, float) in (&*sh_0)
        .iter()
        .zip(&mut *sh_1_x)
        .chain((&*sh_0).iter().zip(&mut *sh_1_y))
        .chain((&*sh_0).iter().zip(&mut *sh_1_z))
    {
        *float /= base;
        if float.is_nan() {
            *float = 0.0;
        }
    }

    let images = &mut [sh_0, sh_1_x, sh_1_y, sh_1_z];

    for (output, image) in std::env::args().skip(3).zip(images.iter_mut()) {
        let floats: &mut [f32] = &mut *image;
        let halfs: &mut [f16] = convert_floats_to_halfs_inline(floats);
        let half_bytes: Vec<u8> = halfs.iter().flat_map(|&half| half.to_le_bytes()).collect();

        let writer = ktx2_tools::Writer {
            header: ktx2_tools::WriterHeader {
                format: Some(ktx2::Format::R16G16B16A16_SFLOAT),
                type_size: 2,
                pixel_width: base_image.width() / 4,
                pixel_height: base_image.height() / num_z_levels,
                pixel_depth: num_z_levels,
                layer_count: 0,
                face_count: 1,
                supercompression_scheme: Some(ktx2::SupercompressionScheme::Zstandard),
            },
            dfd_bytes: &4_u32.to_le_bytes(),
            key_value_pairs: &Default::default(),
            sgd_bytes: &[],
            uncompressed_levels_descending: &[Cow::Owned(half_bytes)],
        };
        writer
            .write(&mut std::fs::File::create(&output).unwrap())
            .unwrap();
    }
}

fn convert_floats_to_halfs_inline(slice: &mut [f32]) -> &mut [f16] {
    let half_slice: &mut [f16] =
        unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut f16, slice.len()) };

    for i in 0..slice.len() {
        let float = slice[i];
        half_slice[i] = f16::from_f32(float);
    }

    half_slice
}
