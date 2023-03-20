use glam::{Vec2, Vec3};
use nalgebra::DVector;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};
use std::path::Path;
#[path = "lightmap-tex-renderer/accessors.rs"]
mod accessors;
use lightmap_tools::{collect_buffer_view_map, NodeTree};
use parking_lot::Mutex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

const EDGE_CONSTRAINTS_WEIGHT: f32 = 5.0;
const COVERED_PIXELS_WEIGHT: f32 = 1.0;
const NONCOVERED_PIXELS_WEIGHT: f32 = 0.1;
const TOLERANCE: f32 = 10.0e-10;
const NUM_ITERATIONS: u32 = 10000;

struct Array2d<T> {
    data: Vec<T>,
    width: u32,
    height: u32,
}

impl<T: Clone> Array2d<T> {
    fn new(width: u32, height: u32, default_t: T) -> Self {
        Self {
            data: vec![default_t; width as usize * height as usize],
            width,
            height,
        }
    }
}

impl<T> Index<(u32, u32)> for Array2d<T> {
    type Output = T;

    fn index(&self, (column, row): (u32, u32)) -> &Self::Output {
        &self.data[row as usize * self.width as usize + column as usize]
    }
}

impl<T> IndexMut<(u32, u32)> for Array2d<T> {
    fn index_mut(&mut self, (column, row): (u32, u32)) -> &mut Self::Output {
        &mut self.data[row as usize * self.width as usize + column as usize]
    }
}

#[derive(Clone, Copy)]
struct HalfEdge {
    a: Vec2,
    b: Vec2,
}

fn uv_to_screen(mut in_vec2: Vec2, w: u32, h: u32) -> Vec2 {
    //in_vec2.y = 1.0 - in_vec2.y;
    in_vec2.x *= w as f32;
    in_vec2.y *= h as f32;
    in_vec2 - Vec2::splat(0.5)
}

fn wrap_coordinate(mut x: i32, size: u32) -> u32 {
    while x < 0 {
        x += size as i32;
    }
    while x >= size as i32 {
        x -= size as i32;
    }
    x as u32
}

struct SeamEdge {
    edges: [HalfEdge; 2],
}

impl SeamEdge {
    fn num_samples(&self, w: u32, h: u32) -> u32 {
        let e0 = uv_to_screen(self.edges[0].b, w, h) - uv_to_screen(self.edges[0].a, w, h);
        let e1 = uv_to_screen(self.edges[1].b, w, h) - uv_to_screen(self.edges[1].a, w, h);
        let len = e0.length().max(e1.length()).max(2.0);
        (len * 3.0) as u32
    }
}

#[derive(Debug, Default)]
struct Mesh {
    indices: Vec<u32>,
    positions: Vec<Vec3>,
    uvs: Vec<Vec2>,
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct HashWrapper<T>(T);

impl<T: PartialEq> Eq for HashWrapper<T> {}

impl Hash for HashWrapper<Vec2> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.0.x.to_bits());
        state.write_u32(self.0.y.to_bits());
    }
}

impl Hash for HashWrapper<Vec3> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.0.x.to_bits());
        state.write_u32(self.0.y.to_bits());
        state.write_u32(self.0.z.to_bits());
    }
}

fn find_seam_edges(mesh: &Mesh) -> Vec<SeamEdge> {
    let mut edge_map: HashMap<
        (HashWrapper<Vec3>, HashWrapper<Vec3>),
        (HashWrapper<Vec2>, HashWrapper<Vec2>),
        _,
    > = HashMap::new();

    let mut seam_edges = Vec::new();

    for tri in mesh.indices.chunks(3) {
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];

        for (index_0, index_1) in edges {
            let v0 = HashWrapper(mesh.positions[index_0 as usize]);
            let v1 = HashWrapper(mesh.positions[index_1 as usize]);
            let uv0 = HashWrapper(mesh.uvs[index_0 as usize]);
            let uv1 = HashWrapper(mesh.uvs[index_1 as usize]);
            //let edge = (v0, v1);
            //let other_edge = (v1, v0);

            let other_edge_key = (v1, v0);

            if !edge_map.contains_key(&other_edge_key) {
                edge_map.insert((v0, v1), (uv0, uv1));
            } else {
                // This edge has already been added once, so we have enough information to see if it's a normal edge, or a "seam edge".
                let (other_uv0, other_uv1) = *edge_map.get(&other_edge_key).unwrap();
                if other_uv0 != uv1 || other_uv1 != uv0 {
                    // UV don't match, so we have a seam
                    let s = SeamEdge {
                        edges: [
                            HalfEdge { a: uv0.0, b: uv1.0 },
                            HalfEdge {
                                a: other_uv1.0,
                                b: other_uv0.0,
                            },
                        ],
                    };
                    seam_edges.push(s);
                }
                edge_map.remove(&other_edge_key); // No longer need this edge, remove it to keep storage low
            }
        }
    }

    seam_edges
}

fn is_inside(x: i32, y: i32, ea: Vec2, eb: Vec2) -> bool {
    x as f32 * (eb.y - ea.y) - y as f32 * (eb.x - ea.x) - ea.x * eb.y + ea.y * eb.x >= 0.0
}

fn rasterize_face(uv0: Vec2, uv1: Vec2, uv2: Vec2, coverage_buf: &mut Array2d<bool>) {
    let uv0 = uv_to_screen(uv0, coverage_buf.width, coverage_buf.height);
    let uv1 = uv_to_screen(uv1, coverage_buf.width, coverage_buf.height);
    let uv2 = uv_to_screen(uv2, coverage_buf.width, coverage_buf.height);

    // Axis aligned bounds of the triangle
    let minx = (uv0.x).min(uv1.x).min(uv2.x) as i32;
    let maxx = (uv0.x).max(uv1.x).max(uv2.x) as i32 + 1;
    let miny = (uv0.y).min(uv1.y).min(uv2.y) as i32;
    let maxy = (uv0.y).max(uv1.y).max(uv2.y) as i32 + 1;

    // The three edges we will test
    let e0a = uv0;
    let e0b = uv1;
    let e1a = uv1;
    let e1b = uv2;
    let e2a = uv2;
    let e2b = uv0;

    // Now just loop over a screen aligned bounding box around the triangle, and test each pixel against all three edges
    for y in miny..maxy {
        for x in minx..maxx {
            if (is_inside(x, y, e0a, e0b) && is_inside(x, y, e1a, e1b) && is_inside(x, y, e2a, e2b))
                || (is_inside(x, y, e0b, e0a)
                    && is_inside(x, y, e1b, e1a)
                    && is_inside(x, y, e2b, e2a))
            {
                let index = (
                    wrap_coordinate(x, coverage_buf.width),
                    wrap_coordinate(y, coverage_buf.height),
                );
                coverage_buf[index] = true;
            }
        }
    }
}

fn dilate_pixel(
    centerx: u32,
    centery: u32,
    image: &image::Rgb32FImage,
    coverage_buf: &Array2d<bool>,
) -> Vec3 {
    let mut num_pixels = 0;
    let mut sum = Vec3::ZERO;
    for yix in centery as i32 - 1..=centery as i32 + 1 {
        for xix in centerx as i32 - 1..=centerx as i32 + 1 {
            let x = wrap_coordinate(xix, image.width());
            let y = wrap_coordinate(yix, image.height());
            if coverage_buf[(x, y)] {
                num_pixels += 1;
                let c = image[(x, y)];
                sum += Vec3::from(c.0);
            }
        }
    }

    if num_pixels > 0 {
        sum / num_pixels as f32
    } else {
        Vec3::ZERO
    }
}

fn calculate_samples_and_weights(pixel_map: &Array2d<i32>, sample: Vec2) -> ([i32; 4], [f32; 4]) {
    let truncu = sample.x as i32;
    let truncv = sample.y as i32;

    let xs = [truncu, truncu + 1, truncu + 1, truncu];
    let ys = [truncv, truncv, truncv + 1, truncv + 1];
    let mut out_ixs = [0; 4];
    for i in 0..4 {
        let x = wrap_coordinate(xs[i], pixel_map.width);
        let y = wrap_coordinate(ys[i], pixel_map.height);
        out_ixs[i] = pixel_map[(x, y)];
    }

    let frac_x = sample.x - truncu as f32;
    let frac_y = sample.y - truncv as f32;
    let out_weights = [
        EDGE_CONSTRAINTS_WEIGHT * (1.0 - frac_x) * (1.0 - frac_y),
        EDGE_CONSTRAINTS_WEIGHT * frac_x * (1.0 - frac_y),
        EDGE_CONSTRAINTS_WEIGHT * frac_x * frac_y,
        EDGE_CONSTRAINTS_WEIGHT * (1.0 - frac_x) * frac_y,
    ];

    (out_ixs, out_weights)
}

fn conjugate_gradient_optimize(
    a: &CsrMatrix<f32>,
    guess: &DVector<f32>,
    b: &DVector<f32>,
    num_iterations: u32,
    tolerance: f32,
) -> DVector<f32> {
    let n = guess.len();
    let mut solution = guess.clone();
    let mut r = b - a * &solution;
    let mut p = r.clone();
    let mut rsq = DVector::dot(&r, &r);
    for i in 0..num_iterations {
        let a_p = a * &p;
        let alpha = rsq / DVector::dot(&p, &a_p);
        solution += alpha * &p;
        r -= alpha * &a_p;
        let rsqnew = DVector::dot(&r, &r);
        if (rsqnew - rsq).abs() < tolerance * n as f32 {
            dbg!(i);
            break;
        }
        let beta = rsqnew / rsq;
        p = &r + beta * &p;
        rsq = rsqnew;
    }

    solution
}

struct PixelInfo {
    x: u32,
    y: u32,
    is_covered: bool,
}

fn compute_pixel_info(
    seam_edges: &[SeamEdge],
    coverage_buf: &Array2d<bool>,
) -> (Vec<PixelInfo>, Array2d<i32>) {
    let w = coverage_buf.width;
    let h = coverage_buf.height;

    let mut pixel_to_pixel_info_map = Array2d::new(w, h, -1);
    let mut pixel_info = Vec::new();

    for s in seam_edges {
        let num_samples = s.num_samples(w, h);
        for e in s.edges {
            let e0 = uv_to_screen(e.a, w, h);
            let e1 = uv_to_screen(e.b, w, h);
            let dt = (e1 - e0) / (num_samples - 1) as f32;
            let mut sample_point = e0;

            for _ in 0..num_samples {
                // Go through the four bilinear sample taps
                let xs = [sample_point.x as u32; 4];
                let ys = [sample_point.y as u32; 4];

                let xs = [sample_point.x as u32, xs[0] + 1, xs[0] + 1, xs[0]];
                let ys = [sample_point.y as u32, ys[0], ys[0] + 1, ys[0] + 1];

                for tap in 0..4 {
                    let x = wrap_coordinate(xs[tap] as i32, w);
                    let y = wrap_coordinate(ys[tap] as i32, h);

                    if pixel_to_pixel_info_map[(x, y)] == -1 {
                        let is_covered = coverage_buf[(x, y)];

                        pixel_info.push(PixelInfo { x, y, is_covered });
                        pixel_to_pixel_info_map[(x, y)] = pixel_info.len() as i32 - 1;
                    }
                }

                sample_point += dt;
            }
        }
    }

    (pixel_info, pixel_to_pixel_info_map)
}

fn join3<A, B, C, RA, RB, RC>(oper_a: A, oper_b: B, oper_c: C) -> (RA, RB, RC)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    C: FnOnce() -> RC + Send,
    RA: Send,
    RB: Send,
    RC: Send,
{
    let ((a, b), c) = rayon::join(|| rayon::join(oper_a, oper_b), oper_c);
    (a, b, c)
}

struct Data {
    a_tb_r: DVector<f32>,
    a_tb_g: DVector<f32>,
    a_tb_b: DVector<f32>,
    initial_guess_r: DVector<f32>,
    initial_guess_g: DVector<f32>,
    initial_guess_b: DVector<f32>,
}

fn setup_ata_matrix(
    seam_edges: &[SeamEdge],
    pixel_info: &[PixelInfo],
    pixel_to_pixel_info_map: &Array2d<i32>,
) -> CsrMatrix<f32> {
    let num_pixels_to_optimise = pixel_info.len();

    let mut matrix_map: HashMap<_, f32> = HashMap::with_capacity(num_pixels_to_optimise);

    let w = pixel_to_pixel_info_map.width;
    let h = pixel_to_pixel_info_map.height;
    for s in seam_edges {
        // Step through the samples of this edge, and compute sample locations for each side of the seam
        let num_samples = s.num_samples(w, h);

        let first_half_edge_start = uv_to_screen(s.edges[0].a, w, h);
        let first_half_edge_end = uv_to_screen(s.edges[0].b, w, h);

        let second_half_edge_start = uv_to_screen(s.edges[1].a, w, h);
        let second_half_edge_end = uv_to_screen(s.edges[1].b, w, h);

        let first_half_edge_step =
            (first_half_edge_end - first_half_edge_start) / (num_samples - 1) as f32;
        let second_half_edge_step =
            (second_half_edge_end - second_half_edge_start) / (num_samples - 1) as f32;

        let mut first_half_edge_sample = first_half_edge_start;
        let mut second_half_edge_sample = second_half_edge_start;
        for _ in 0..num_samples {
            // Sample locations for the two corresponding sets of sample points
            let (first_half_edge, first_half_edge_weights) =
                calculate_samples_and_weights(pixel_to_pixel_info_map, first_half_edge_sample);
            let (second_half_edge, second_half_edge_weights) =
                calculate_samples_and_weights(pixel_to_pixel_info_map, second_half_edge_sample);

            /*
            Now, compute the covariance for the difference of these two vectors.
            If a is the first vector (first half edge) and b is the second, then we compute the covariance, without
            intermediate storage, like so:
            (a-b)*(a-b)^t = a*a^t + b*b^t - a*b^t-b*a^t
            */
            for i in 0..4 {
                for j in 0..4 {
                    // + a*a^t
                    *matrix_map
                        .entry((first_half_edge[i], first_half_edge[j]))
                        .or_default() += first_half_edge_weights[i] * first_half_edge_weights[j];
                    // + b*b^t
                    *matrix_map
                        .entry((second_half_edge[i], second_half_edge[j]))
                        .or_default() += second_half_edge_weights[i] * second_half_edge_weights[j];

                    // - a*b^t
                    *matrix_map
                        .entry((first_half_edge[i], second_half_edge[j]))
                        .or_default() -= first_half_edge_weights[i] * second_half_edge_weights[j];

                    // - b*a^t
                    *matrix_map
                        .entry((second_half_edge[i], first_half_edge[j]))
                        .or_default() -= second_half_edge_weights[i] * first_half_edge_weights[j];
                }
            }

            first_half_edge_sample += first_half_edge_step;
            second_half_edge_sample += second_half_edge_step;
        }
    }

    for (i, pi) in pixel_info.iter().enumerate() {
        // Set up equality cost, trying to keep the pixel at its original value
        // Note: for non-covered pixels the weight is much lower, since those are the pixels
        // we primarily want to modify (we'll want to keep it >0 though, to reduce the risk
        // of extreme values that can't fit in 8 bit color channels)
        let weight = if pi.is_covered {
            COVERED_PIXELS_WEIGHT
        } else {
            NONCOVERED_PIXELS_WEIGHT
        };

        *matrix_map.entry((i as i32, i as i32)).or_default() += weight;
    }

    let mut coo_matrix = CooMatrix::new(pixel_info.len(), pixel_info.len());

    for (&(x, y), &value) in matrix_map.iter() {
        coo_matrix.push(x as usize, y as usize, value);
    }

    CsrMatrix::from(&coo_matrix)
}

fn setup_least_squares(
    pixel_info: &[PixelInfo],
    image: &image::Rgb32FImage,
    coverage_buf: &Array2d<bool>,
) -> Data {
    let num_pixels_to_optimise = pixel_info.len();

    let mut a_tb_r = DVector::zeros(num_pixels_to_optimise);
    let mut a_tb_g = DVector::zeros(num_pixels_to_optimise);
    let mut a_tb_b = DVector::zeros(num_pixels_to_optimise);
    let mut initial_guess_r = DVector::zeros(num_pixels_to_optimise);
    let mut initial_guess_g = DVector::zeros(num_pixels_to_optimise);
    let mut initial_guess_b = DVector::zeros(num_pixels_to_optimise);

    for (i, pi) in pixel_info.iter().enumerate() {
        // Set up equality cost, trying to keep the pixel at its original value
        // Note: for non-covered pixels the weight is much lower, since those are the pixels
        // we primarily want to modify (we'll want to keep it >0 though, to reduce the risk
        // of extreme values that can't fit in 8 bit color channels)
        let weight = if pi.is_covered {
            COVERED_PIXELS_WEIGHT
        } else {
            NONCOVERED_PIXELS_WEIGHT
        };

        let colour = if pi.is_covered {
            Vec3::from(image[(pi.x, pi.y)].0)
        } else {
            dilate_pixel(pi.x, pi.y, image, coverage_buf)
        };

        // Set up the three right hand sides (one for R, G, and B).
        // Note AtRHS represents the transpose of the system matrix A multiplied by the RHS
        a_tb_r[i] = colour.x * weight;
        a_tb_g[i] = colour.y * weight;
        a_tb_b[i] = colour.z * weight;

        // Set up the initial guess for the solution.
        initial_guess_r[i] = colour.x;
        initial_guess_g[i] = colour.y;
        initial_guess_b[i] = colour.z;
    }

    Data {
        a_tb_r,
        a_tb_g,
        a_tb_b,
        initial_guess_r,
        initial_guess_g,
        initial_guess_b,
    }
}

fn main() -> anyhow::Result<()> {
    let mesh_filename = std::env::args().nth(1).unwrap();
    let tex_filename = std::env::args().nth(2).unwrap();
    let tex_out_filename = std::env::args().nth(3).unwrap();

    let mesh = {
        let bytes = std::fs::read(&mesh_filename).unwrap();

        let (gltf, buffer): (
            goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
            _,
        ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();

        let node_tree = NodeTree::new(&gltf);
        let buffer_view_map = collect_buffer_view_map(&gltf, buffer, Path::new(&mesh_filename))?;

        let mut combined_positions = Vec::new();
        let mut combined_indices = Vec::new();
        let mut combined_second_uvs = Vec::new();

        for (node_id, node) in gltf.nodes.iter().enumerate() {
            let transform = node_tree.transform_of(node_id);

            let mesh_id = match node.mesh {
                Some(mesh_id) => mesh_id,
                None => continue,
            };

            let mesh = &gltf.meshes[mesh_id];

            for (primitive_id, primitive) in mesh.primitives.iter().enumerate() {
                let reader = accessors::PrimitiveReader::new(&gltf, primitive, &buffer_view_map);

                let positions = match reader.read_positions()? {
                    None => {
                        println!(
                            "Positions missing for mesh {}, primitive {}. Skipping",
                            mesh_id, primitive_id
                        );
                        continue;
                    }
                    Some(positions) => positions,
                };

                let indices = match reader.read_indices()? {
                    None => {
                        println!(
                            "Indices missing for mesh {}, primitive {}. Skipping",
                            mesh_id, primitive_id
                        );
                        continue;
                    }
                    Some(indices) => indices,
                };

                let second_uvs = match reader.read_second_uvs()? {
                    None => {
                        println!(
                            "Second UVs missing for mesh {}, primitive {}. Skipping",
                            mesh_id, primitive_id
                        );
                        continue;
                    }
                    Some(second_uvs) => second_uvs,
                };

                let indices_offset = combined_positions.len() as u32;

                combined_indices.extend(indices.iter().map(|index| indices_offset + index));
                combined_positions.extend(
                    positions
                        .iter()
                        .map(|position| (transform * position.extend(1.0)).truncate()),
                );
                combined_second_uvs.extend_from_slice(&second_uvs);
            }
        }

        Mesh {
            positions: combined_positions,
            indices: combined_indices,
            uvs: combined_second_uvs,
        }
    };

    let mut orig_image = image::open(&tex_filename).unwrap();

    let w = orig_image.width() as usize / 4;
    let h = orig_image.height() as usize;

    let seam_edges = find_seam_edges(&mesh);
    dbg!(&seam_edges.len());

    let mut coverage_buf = Array2d::new(w as u32, h as u32, false);
    for tri in mesh.indices.chunks(3) {
        let uv0 = mesh.uvs[tri[0] as usize];
        let uv1 = mesh.uvs[tri[1] as usize];
        let uv2 = mesh.uvs[tri[2] as usize];

        rasterize_face(uv0, uv1, uv2, &mut coverage_buf);
    }

    let (pixel_info, pixel_to_pixel_info_map) = compute_pixel_info(&seam_edges, &coverage_buf);
    let num_pixels_to_optimise = pixel_info.len();

    let a_t_a = setup_ata_matrix(&seam_edges, &pixel_info, &pixel_to_pixel_info_map);

    dbg!(&num_pixels_to_optimise);

    let images = [
        orig_image
            .crop(0, 0, orig_image.width() / 4, orig_image.height())
            .to_rgb32f(),
        orig_image
            .crop(
                orig_image.width() / 4,
                0,
                orig_image.width() / 4,
                orig_image.height(),
            )
            .to_rgb32f(),
        orig_image
            .crop(
                orig_image.width() / 2,
                0,
                orig_image.width() / 4,
                orig_image.height(),
            )
            .to_rgb32f(),
        orig_image
            .crop(
                orig_image.width() * 3 / 4,
                0,
                orig_image.width() / 4,
                orig_image.height(),
            )
            .to_rgb32f(),
    ];

    let orig_image = Mutex::new(orig_image.to_rgb32f());

    images.par_iter().enumerate().for_each(|(i, image)| {
        let data = setup_least_squares(&pixel_info, image, &coverage_buf);

        dbg!(());

        let (solution_r, solution_g, solution_b) = join3(
            || {
                conjugate_gradient_optimize(
                    &a_t_a,
                    &data.initial_guess_r,
                    &data.a_tb_r,
                    NUM_ITERATIONS,
                    TOLERANCE,
                )
            },
            || {
                conjugate_gradient_optimize(
                    &a_t_a,
                    &data.initial_guess_g,
                    &data.a_tb_g,
                    NUM_ITERATIONS,
                    TOLERANCE,
                )
            },
            || {
                conjugate_gradient_optimize(
                    &a_t_a,
                    &data.initial_guess_b,
                    &data.a_tb_b,
                    NUM_ITERATIONS,
                    TOLERANCE,
                )
            },
        );

        let offset = i as u32 * image.width();

        let mut orig_image = orig_image.lock();

        for (i, pi) in pixel_info.iter().enumerate() {
            orig_image[(offset + pi.x, pi.y)].0 = [solution_r[i], solution_g[i], solution_b[i]];
        }
    });

    orig_image.lock().save(&tex_out_filename).unwrap();

    Ok(())
}
