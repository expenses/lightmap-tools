#[path = "lightmap-tex-renderer/accessors.rs"]
mod accessors;
use goth_gltf::default_extensions::Extensions;
use goth_gltf::extensions::CompressionMode;
use std::collections::HashMap;
use std::path::Path;
use wgpu::util::DeviceExt;
use lightmap_tools::{NodeTree, collect_buffer_view_map};

use structopt::StructOpt;

#[derive(StructOpt)]
pub struct Opts {
    filepath: std::path::PathBuf,
    width: u32,
    height: u32,
}

fn main() -> anyhow::Result<()> {
    let opts = Opts::from_args();

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

            for triangle in indices.chunks(3) {
                let indices: [usize; 3] = std::array::from_fn(|i| triangle[i] as usize);

                let positions: [glam::Vec3; 3] =
                    indices.map(|index| (transform * positions[index].extend(1.0)).truncate());

                //let normal = //(positions[1] - positions[0]).cross(positions[2] - positions[0]).normalize();

                combined_positions.extend_from_slice(&positions);
                combined_normals.extend_from_slice(&indices.map(|index| normals[index]));
                combined_second_uvs.extend_from_slice(&indices.map(|index| second_uvs[index]));
            }
        }
    }

    let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    let instance = wgpu::Instance::new(backend);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("No suitable GPU adapters found on the system!");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("device"),
            features: adapter.features(),
            limits: adapter.limits(),
        },
        None,
    ))
    .expect("Unable to find a suitable GPU adapter!");

    let num_positions = combined_positions.len() as u32;

    let positions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        contents: bytemuck::cast_slice(&combined_positions),
        usage: wgpu::BufferUsages::STORAGE,
        label: None,
    });

    let normals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        contents: bytemuck::cast_slice(&combined_normals),
        usage: wgpu::BufferUsages::STORAGE,
        label: None,
    });

    let second_uvs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        contents: bytemuck::cast_slice(&combined_second_uvs),
        usage: wgpu::BufferUsages::STORAGE,
        label: None,
    });

    let storage_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        count: None,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    };

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[storage_entry(0), storage_entry(1), storage_entry(2)],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: positions_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: normals_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: second_uvs_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader =
        device.create_shader_module(wgpu::include_wgsl!("lightmap-tex-renderer/shader.wgsl"));

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vertex",
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            conservative: false,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fragment",
            targets: &[
                Some(wgpu::TextureFormat::Rgba32Float.into()),
                Some(wgpu::TextureFormat::Rgba32Float.into()),
            ],
        }),
        multiview: None,
    });

    let output_dim = wgpu::Extent3d {
        width: opts.width * 4,
        height: opts.height * 4,
        depth_or_array_layers: 1,
    };

    let calculate_padding_for_pixel_size = |pixel_size_in_bytes| {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u32 / pixel_size_in_bytes;
        let padded_width_padding = (align - output_dim.width % align) % align;
        output_dim.width + padded_width_padding
    };

    let buffer_descriptor_for_pixel_size = |pixel_size_in_bytes| wgpu::BufferDescriptor {
        size: (calculate_padding_for_pixel_size(pixel_size_in_bytes) * output_dim.height * pixel_size_in_bytes) as u64,
        label: None,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    };

    let positions_output_buffer = device.create_buffer(&buffer_descriptor_for_pixel_size(4 * 4));
    let normals_output_buffer = device.create_buffer(&buffer_descriptor_for_pixel_size(4 * 4));

    let texture_descriptor_for_format = |format| wgpu::TextureDescriptor {
        label: None,
        size: output_dim,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    };

    let positions_tex = device.create_texture(&texture_descriptor_for_format(wgpu::TextureFormat::Rgba32Float));
    let normals_tex = device.create_texture(&texture_descriptor_for_format(wgpu::TextureFormat::Rgba32Float));

    let positions_tex_view = positions_tex.create_view(&Default::default());
    let normals_tex_view = normals_tex.create_view(&Default::default());

    let mut command_encoder = device.create_command_encoder(&Default::default());

    let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[
            Some(wgpu::RenderPassColorAttachment {
                view: &positions_tex_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &normals_tex_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            }),
        ],
        depth_stencil_attachment: None,
    });

    render_pass.set_pipeline(&pipeline);
    render_pass.set_bind_group(0, &bind_group, &[]);

    render_pass.draw(0..num_positions, 0..1);

    drop(render_pass);

    let image_layout_for_pixel_size = |pixel_size_in_bytes| wgpu::ImageDataLayout {
        offset: 0,
        bytes_per_row: Some(std::num::NonZeroU32::new(calculate_padding_for_pixel_size(pixel_size_in_bytes) * pixel_size_in_bytes).unwrap()),
        rows_per_image: None,
    };

    command_encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &positions_tex,
            mip_level: 0,
            origin: Default::default(),
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &positions_output_buffer,
            layout: image_layout_for_pixel_size(4 * 4),
        },
        output_dim,
    );

    command_encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &normals_tex,
            mip_level: 0,
            origin: Default::default(),
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &normals_output_buffer,
            layout: image_layout_for_pixel_size(4 * 4),
        },
        output_dim,
    );

    queue.submit(Some(command_encoder.finish()));

    let (positions_floats, normals_floats) = pollster::block_on(futures::future::join(
        slice_to_bytes(
            &positions_output_buffer.slice(..),
            &device,
            output_dim,
            calculate_padding_for_pixel_size(4 * 4),
        ),
        slice_to_bytes(
            &normals_output_buffer.slice(..),
            &device,
            output_dim,
            calculate_padding_for_pixel_size(4 * 4),
        ),
    ));

    image::Rgba32FImage::from_raw(output_dim.width, output_dim.height, positions_floats)
        .unwrap()
        .save("positions.exr")
        .unwrap();
    image::Rgba32FImage::from_raw(output_dim.width, output_dim.height, normals_floats)
        .unwrap()
        .save("normals.exr")
        .unwrap();

    Ok(())
}

async fn slice_to_bytes(
    buffer_slice: &wgpu::BufferSlice<'_>,
    device: &wgpu::Device,
    extent: wgpu::Extent3d,
    padded_width: u32,
) -> Vec<f32> {
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::Wait);

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let pixels: &[glam::Vec4] = bytemuck::cast_slice(&data);

        let mut output = Vec::new();

        for row in 0..extent.height {
            let offset = (row * padded_width) as usize;

            output.extend_from_slice(bytemuck::cast_slice(
                &pixels[offset..offset + extent.width as usize],
            ));
        }

        output
    } else {
        panic!()
    }
}
