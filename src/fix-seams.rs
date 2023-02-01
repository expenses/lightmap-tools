#[path = "lightmap-tex-renderer/accessors.rs"]
mod accessors;
//mod bindings;
use glam::{UVec2, Vec2, Vec3};
use goth_gltf::default_extensions::Extensions;
use goth_gltf::extensions::CompressionMode;
use lightmap_tools::{collect_buffer_view_map, NodeTree};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use wgpu::util::DeviceExt;

use structopt::StructOpt;

#[derive(StructOpt)]
pub struct Opts {
    filepath: std::path::PathBuf,
    image: std::path::PathBuf,
    width: u32,
    height: u32,
}

fn main() -> anyhow::Result<()> {
    let opts = Opts::from_args();

    let mut img = image::open(&opts.image).unwrap();

    let mut img = img
        //.crop_imm(0, 0, img.width() / 4, img.height())
        .to_rgb32f();

    let bytes = std::fs::read(&opts.filepath).unwrap();

    let (gltf, buffer): (
        goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>,
        _,
    ) = goth_gltf::Gltf::from_bytes(&bytes).unwrap();

    let node_tree = NodeTree::new(&gltf);
    let buffer_view_map = collect_buffer_view_map(&gltf, buffer, &opts.filepath)?;

    let mut combined_positions = Vec::new();
    let mut combined_normals = Vec::new();
    let mut combined_second_uvs = Vec::new();
    let mut combined_indices = Vec::new();

    for (node_id, node) in gltf.nodes.iter().enumerate() {
        let transform = node_tree.transform_of(node_id);
        let normal_matrix = glam::Mat3::from_mat4(transform.inverse().transpose());

        let mesh_id = match node.mesh {
            Some(mesh_id) => mesh_id,
            None => continue,
        };

        let mesh = &gltf.meshes[mesh_id];

        for (primitive_id, primitive) in mesh.primitives.iter().enumerate() {
            let reader = accessors::PrimitiveReader::new(&gltf, &primitive, &buffer_view_map);

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

            let normals = reader.read_normals()?.unwrap();

            let indices_offset = combined_positions.len() as u32;

            let index_offset = combined_positions.len() as u32;

            combined_indices.extend(indices.iter().map(|&index| index + index_offset));
            combined_positions.extend(
                positions
                    .iter()
                    .map(|&position| (transform * position.extend(1.0)).truncate()),
            );
            combined_normals.extend(normals.iter().map(|&normal| normal_matrix * normal));
            combined_second_uvs.extend_from_slice(&second_uvs);
        }
    }

    let mut edge_indices: Vec<(u32, u32)> = Vec::new();

    let mut tree: rstar::RTree<Edge> = Default::default();

    let mut contained: HashSet<(u32, u32)> = HashSet::default();

    let edge_iterator = combined_indices
        .chunks(3)
        .flat_map(|tri| [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]);

    for (a, b) in edge_iterator.clone() {
        let edge = Edge::new(
            Vertex {
                pos: combined_positions[a as usize],
                normal: combined_normals[a as usize],
                uv: combined_second_uvs[a as usize],
                index: a,
            },
            Vertex {
                pos: combined_positions[b as usize],
                normal: combined_normals[b as usize],
                uv: combined_second_uvs[b as usize],
                index: b,
            },
        );
        if !contained.contains(&(edge.a.index, edge.b.index)) {
            tree.insert(edge);
        }
        contained.insert((edge.a.index, edge.b.index));
    }

    let lookup_distance = 0.001;
    let lookup_distance_sq = lookup_distance * lookup_distance;

    let uv_lookup_distance = 0.0001;
    let uv_lookup_distance_sq = lookup_distance * lookup_distance;

    let mut matches = 0;

    use svg::node::element::path::Data;
    use svg::node::element::*;
    use svg::Document;

    let mut doc = svg::Document::new();

    let mut data = Data::new();
    let mut circles = Vec::new();

    let scale = glam::Vec2::new(opts.width as f32, opts.height as f32);

    let mut stitching_points_to_match = Vec::new();

    let mut total_stitching_points = 0;

    for (a, b) in edge_iterator {
        let compare_edge = Edge::new(
            Vertex {
                pos: combined_positions[a as usize],
                normal: combined_normals[a as usize],
                uv: combined_second_uvs[a as usize],
                index: a,
            },
            Vertex {
                pos: combined_positions[b as usize],
                normal: combined_normals[b as usize],
                uv: combined_second_uvs[b as usize],
                index: b,
            },
        );

        data = data
            .move_to(<(f32, f32)>::from(compare_edge.a.uv * scale))
            .line_to(<(f32, f32)>::from(compare_edge.b.uv * scale));

        for x in tree.locate_within_distance(compare_edge.a.pos.into(), lookup_distance_sq) {
            if x.a.index == compare_edge.a.index && x.b.index == compare_edge.b.index {
                continue;
            }

            if x.b.pos.distance_squared(compare_edge.b.pos) > lookup_distance_sq {
                continue;
            }

            if x.a.uv.distance_squared(compare_edge.a.uv) < uv_lookup_distance_sq {
                continue;
            }

            if x.b.uv.distance_squared(compare_edge.b.uv) < uv_lookup_distance_sq {
                continue;
            }

            let x_normal = (x.a.normal + x.b.normal).normalize();

            let edge_normal = (compare_edge.a.normal + compare_edge.b.normal).normalize();

            if x_normal.dot(edge_normal) < 0.9 {
                continue;
            }

            let x_center = (x.a.uv + x.b.uv) / 2.0;
            let comp_center = (compare_edge.a.uv + compare_edge.b.uv) / 2.0;

            circles.push(
                Circle::new()
                    .set("fill", "none")
                    .set("stroke", "blue")
                    .set("stroke-width", 0.2)
                    .set("cx", comp_center.x * scale.x)
                    .set("cy", comp_center.y * scale.y)
                    .set("r", 0.5),
            );

            circles.push(
                Circle::new()
                    .set("fill", "none")
                    .set("stroke", "red")
                    .set("stroke-width", 0.2)
                    .set("cx", x_center.x * scale.x)
                    .set("cy", x_center.y * scale.y)
                    .set("r", 0.5),
            );

            let edge_uv_length = (compare_edge.a.uv * scale).distance(compare_edge.b.uv * scale);
            let x_uv_length = (x.a.uv * scale).distance(x.b.uv * scale);

            let max_edge_length = edge_uv_length.max(x_uv_length);

            // the paper uses 3 per texel but that might be too many for testing.
            let num_sample_points_per_pixel_length = 0.5; //1.0;//0.005;//3.;

            let num_sample_points =
                (max_edge_length * num_sample_points_per_pixel_length).ceil() as usize;

            let mut compare_sample_points: Vec<Vec2> = Vec::with_capacity(num_sample_points);
            let mut x_sample_points: Vec<Vec2> = Vec::with_capacity(num_sample_points);

            let step = 1.0 / (num_sample_points as f32 + 1.0);

            let compare_diff = (compare_edge.b.uv - compare_edge.a.uv) * scale;
            let x_diff = (x.b.uv - x.a.uv) * scale;

            for i in 1..1 + num_sample_points {
                compare_sample_points
                    .push(compare_edge.a.uv * scale + compare_diff * step * i as f32);
                x_sample_points.push(x.a.uv * scale + x_diff * step * i as f32);

                let wwww = compare_edge.a.uv * scale + compare_diff * step * i as f32; //x.a.uv * scale + x_diff * step * i as f32;

                let zzzz = x.a.uv * scale + x_diff * step * i as f32;

                circles.push(
                    Circle::new()
                        .set("fill", "none")
                        .set("stroke", "green")
                        .set("stroke-width", 0.2)
                        .set("cx", zzzz.x)
                        .set("cy", zzzz.y)
                        .set("r", 0.25),
                );

                circles.push(
                    Circle::new()
                        .set("fill", "none")
                        .set("stroke", "green")
                        .set("stroke-width", 0.2)
                        .set("cx", wwww.x)
                        .set("cy", wwww.y)
                        .set("r", 0.25),
                );
            }

            total_stitching_points += num_sample_points;

            stitching_points_to_match.push((compare_sample_points, x_sample_points));;
        }

        tree.remove(&compare_edge);
    }

    let mut img2 = img.clone();

    for (edge_a_points, edge_b_points) in &stitching_points_to_match {
        for (&a, &b) in edge_a_points.iter().zip(edge_b_points) {
            let (a_coords, a_weights) =
                get_coords_and_weights(a, UVec2::new(opts.width, opts.height));
            let (b_coords, b_weights) =
                get_coords_and_weights(b, UVec2::new(opts.width, opts.height));

            let mut a_value = Vec3::ZERO;
            for (&coord, weight) in a_coords.iter().zip(a_weights) {
                a_value += Vec3::from(img[coord].0) * weight;
            }

            let mut b_value = Vec3::ZERO;
            for (&coord, weight) in b_coords.iter().zip(b_weights) {
                b_value += Vec3::from(img[coord].0) * weight;
            }

            let avg = (a_value + b_value) / 2.0;

            //dbg!(a_weights, b_weights);

            for coord in a_coords.into_iter().chain(b_coords) {
                //img2[coord] = image::Rgb(avg.into());
            }
        }
    }

    doc = doc.add(Image::new().set("href", "img3.png"));

    let path = Path::new()
        .set("fill", "none")
        .set("stroke", "blue")
        .set("stroke-width", 0.1)
        .set("d", data);

    doc = doc.add(path);

    image::DynamicImage::from(img2)
        .to_rgb8()
        .save("img3.png")
        .unwrap();

    for circle in circles {
        doc = doc.add(circle);
    }

    doc = doc.set("viewBox", (0, 0, opts.width, opts.height));

    svg::save("image.svg", &doc).unwrap();

    dbg!(
        &stitching_points_to_match.len(),
        total_stitching_points,
        total_stitching_points as f32 / stitching_points_to_match.len() as f32
    );

    Ok(())
}

fn get_coords_and_weights(p: Vec2, dimensions: UVec2) -> ([(u32, u32); 4], [f32; 4]) {
    let p = p - 0.5;

    let p_fract = p.fract();
    let p_u = p.as_uvec2();
    let p_u_1 = (p_u + 1).min(dimensions - 1);

    (
        [
            (p_u.x, p_u.y),
            (p_u_1.x, p_u.y),
            (p_u.x, p_u_1.y),
            (p_u_1.x, p_u_1.y),
        ],
        [
            (1.0 - p_fract.x) * (1.0 - p_fract.y),
            p_fract.x * (1.0 - p_fract.y),
            (1.0 - p_fract.x) * p_fract.y,
            p_fract.x * p_fract.y,
        ],
    )
}

struct SeamEdge {
    a: Vec2,
    b: Vec2,
    stitching_points: Vec<Vec2>,
}

fn compare_vecs(a: Vec3, b: Vec3) -> std::cmp::Ordering {
    match a.x.partial_cmp(&b.x) {
        Some(ordering @ (std::cmp::Ordering::Less | std::cmp::Ordering::Greater)) => {
            return ordering
        }
        _ => {}
    }

    match a.y.partial_cmp(&b.y) {
        Some(ordering @ (std::cmp::Ordering::Less | std::cmp::Ordering::Greater)) => {
            return ordering
        }
        _ => {}
    }

    a.z.partial_cmp(&b.z).unwrap()
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
struct Vertex {
    pos: Vec3,
    uv: Vec2,
    normal: Vec3,
    index: u32,
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
struct Edge {
    a: Vertex,
    b: Vertex,
}

impl From<Vec3> for Edge {
    fn from(vec: Vec3) -> Self {
        Self {
            a: Vertex {
                pos: vec,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

impl Edge {
    fn new(mut a: Vertex, mut b: Vertex) -> Self {
        if compare_vecs(a.pos, b.pos) == std::cmp::Ordering::Greater {
            std::mem::swap(&mut a, &mut b);
        }

        Self { a, b }
    }
}

impl rstar::Point for Edge {
    type Scalar = f32;

    const DIMENSIONS: usize = 3;

    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Self::from(Vec3::new(generator(0), generator(1), generator(2)))
    }
    fn nth(&self, index: usize) -> Self::Scalar {
        self.a.pos[index]
    }
    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        &mut self.a.pos[index]
    }
}
