use std::io::Write;
use structopt::StructOpt;

#[derive(StructOpt)]
pub struct Opts {
    path: std::path::PathBuf,
    output: std::path::PathBuf,
    n_x: u32,
    n_y: u32,
    n_z: u32,
}

struct ProbeArray {
    image: image::ImageBuffer<image::Rgb<f32>, Vec<f32>>,
    width: u32,
    height: u32,
    depth: u32,
}

impl ProbeArray {
    fn get_coefficients(&self, x: u32, mut y: u32, z: u32) -> [glam::Vec3; 4] {
        y += self.height * z;

        [
            glam::Vec3::from(self.image.get_pixel(x, y).0),
            glam::Vec3::from(self.image.get_pixel(x + self.width, y).0),
            glam::Vec3::from(self.image.get_pixel(x + self.width * 2, y).0),
            glam::Vec3::from(self.image.get_pixel(x + self.width * 3, y).0),
        ]
    }

    fn try_get(&self, x: u32, y: u32, z: u32) -> Option<[glam::Vec3; 4]> {
        if z >= self.depth || x >= self.width || y >= self.height {
            return None;
        }

        let coefs = self.get_coefficients(x, y, z);

        //dbg!(&coefs);

        if coefs[0].length() < 0.01 {
            return None;
        }

        Some(self.get_coefficients(x, y, z))
    }

    fn insert(&mut self, x: u32, mut y: u32, z: u32, coefs: [glam::Vec3; 4]) {
        y += self.height * z;

        self.image.put_pixel(x, y, image::Rgb(coefs[0].into()));
        self.image
            .put_pixel(x + self.width, y, image::Rgb(coefs[1].into()));
        self.image
            .put_pixel(x + self.width * 2, y, image::Rgb(coefs[2].into()));
        self.image
            .put_pixel(x + self.width * 3, y, image::Rgb(coefs[3].into()));
    }
}

fn add(a: [glam::Vec3; 4], b: [glam::Vec3; 4]) -> [glam::Vec3; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn div(a: [glam::Vec3; 4], b: f32) -> [glam::Vec3; 4] {
    [a[0] / b, a[1] / b, a[2] / b, a[3] / b]
}

fn iter_neighbours(base_x: u32, base_y: u32, base_z: u32) -> impl Iterator<Item = (u32, u32, u32)> {
    (base_x.saturating_sub(1)..=base_x.saturating_add(1))
        .flat_map(move |x| {
            (base_y.saturating_sub(1)..=base_y.saturating_add(1)).map(move |y| (x, y))
        })
        .flat_map(move |(x, y)| {
            (base_z.saturating_sub(1)..=base_z.saturating_add(1)).map(move |z| (x, y, z))
        })
        .filter(move |&coord| coord != (base_x, base_y, base_z))
}

fn main() {
    let opts = Opts::from_args();

    // layer/coef/values
    let mut coefficients: [Vec<glam::Vec4>; 4] = [vec![], vec![], vec![], vec![]];

    let mut array = ProbeArray {
        image: (to_image_buffer(image::open(opts.path).unwrap())),
        width: opts.n_x,
        height: opts.n_y,
        depth: opts.n_z,
    };

    let mut check_for_invalid = true;
    let mut invalid_list = Vec::new();

    // Bound the number of times we check
    for _ in 0..opts.n_x.max(opts.n_y).max(opts.n_z) {
        if !check_for_invalid {
            break;
        }

        for z in 0..opts.n_z {
            for y in 0..opts.n_y {
                'inner: for x in 0..opts.n_x {
                    let cubemap_coefficients = array.get_coefficients(x, y, z);

                    if cubemap_coefficients[0] != glam::Vec3::ZERO {
                        continue 'inner;
                    }

                    let mut neighbour_sum = [glam::Vec3::ZERO; 4];
                    let mut num_valid_neighbours = 0;

                    for (x, y, z) in iter_neighbours(x, y, z) {
                        if let Some(coefs) = array.try_get(x, y, z) {
                            neighbour_sum = add(neighbour_sum, coefs);
                            num_valid_neighbours += 1
                        }
                    }

                    if num_valid_neighbours == 0 {
                        continue 'inner;
                    }

                    invalid_list.push((x, y, z, div(neighbour_sum, num_valid_neighbours as f32)));
                }
            }
        }

        check_for_invalid = !invalid_list.is_empty();

        dbg!(&invalid_list.len());

        for (x, y, z, replacement) in invalid_list.drain(..) {
            array.insert(x, y, z, replacement);
        }
    }

    //dbg!(array.get_coefficients(7, 24, 101), array.get_coefficients(7, 24, 101) == [glam::Vec3::ZERO; 4]);

    for z in 0..opts.n_z {
        for y in 0..opts.n_y {
            for x in 0..opts.n_x {
                let mut cubemap_coefficients = array.get_coefficients(x, y, z);

                for i in 1..4 {
                    for j in 0..3 {
                        if cubemap_coefficients[0][j] == 0.0 {
                            cubemap_coefficients[i][j] = 0.0;
                        } else {
                            cubemap_coefficients[i][j] /= cubemap_coefficients[0][j]
                        }
                    }
                }

                for i in 0..4 {
                    coefficients[i].push(cubemap_coefficients[i].extend(1.0));
                }
            }
        }
    }

    for (i, coef) in coefficients.iter().enumerate() {
        let mut file = std::fs::File::create(opts.output.join(&format!("{}.bin", i))).unwrap();
        file.write_all(bytemuck::cast_slice(coef)).unwrap();
    }
}

fn to_image_buffer(image: image::DynamicImage) -> image::ImageBuffer<image::Rgb<f32>, Vec<f32>> {
    image.to_rgb32f()
}
