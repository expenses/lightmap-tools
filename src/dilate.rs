#[path = "lightmap-tex-renderer/accessors.rs"]
mod accessors;
use goth_gltf::default_extensions::Extensions;
use goth_gltf::extensions::CompressionMode;
use std::collections::HashMap;
use std::path::Path;
use wgpu::util::DeviceExt;

fn main() -> anyhow::Result<()> {
    let filepath = std::env::args().nth(1).unwrap();
    let image = image::open(&filepath).unwrap();

    let image = image.to_rgba32f();

    let (device, queue) = lightmap_tools::init_wgpu()?;

    let extent = wgpu::Extent3d {
        width: image.width(),
        height: image.height(),
        depth_or_array_layers: 1,
    };

    let texture_descriptor = wgpu::TextureDescriptor {
        label: None,
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
    };

    let texture =
        device.create_texture_with_data(&queue, &texture_descriptor, bytemuck::cast_slice(&*image));
    let texture_view = texture.create_view(&Default::default());

    let output_texture = device.create_texture(&texture_descriptor);
    let output_texture_view = output_texture.create_view(&Default::default());

    let pixel_size_in_bytes = 4 * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u32 / pixel_size_in_bytes;
    let padded_width_padding = (align - extent.width % align) % align;
    let padded_width = extent.width + padded_width_padding;

    let buffer_descriptor = wgpu::BufferDescriptor {
        size: (padded_width * extent.height * 4 * 4) as u64,
        label: None,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    };

    let output_buffer = device.create_buffer(&buffer_descriptor);

    let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        count: None,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::ReadWrite,
            format: wgpu::TextureFormat::Rgba32Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
    };

    let buffer_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        count: None,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[extent.width as i32 / 4]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[texture_entry(0), texture_entry(1), buffer_entry(2)],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&output_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::include_wgsl!("dilate.wgsl"));

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "compute",
    });

    let mut command_encoder = device.create_command_encoder(&Default::default());

    let mut compute_pass =
        command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    fn next_multiple(value: u32) -> u32 {
        (((value - 1) / 8) + 1) * 8
    }

    compute_pass.dispatch_workgroups(
        next_multiple(extent.width / 4),
        next_multiple(extent.height),
        1,
    );

    drop(compute_pass);

    let image_layout = wgpu::ImageDataLayout {
        offset: 0,
        bytes_per_row: Some(std::num::NonZeroU32::new(padded_width * 4 * 4).unwrap()),
        rows_per_image: None,
    };

    command_encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &output_texture,
            mip_level: 0,
            origin: Default::default(),
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: image_layout,
        },
        extent,
    );

    /*
    let image_layout = wgpu::ImageDataLayout {
        offset: 0,
        bytes_per_row: Some(std::num::NonZeroU32::new(padded_width * 4 * 4).unwrap()),
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
            layout: image_layout,
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
            layout: image_layout,
        },
        output_dim,
    );*/

    queue.submit(Some(command_encoder.finish()));

    let floats = slice_to_bytes(&output_buffer.slice(..), &device, extent, padded_width);

    image::Rgba32FImage::from_raw(extent.width, extent.height, floats)
        .unwrap()
        .save("floats.exr")
        .unwrap();

    /*
    let positions_floats = slice_to_bytes(&positions_output_buffer.slice(..), &device, output_dim, padded_width);
    let normals_floats = slice_to_bytes(&normals_output_buffer.slice(..), &device, output_dim, padded_width);

    /*let positions_output_buffer_slice = positions_output_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    positions_output_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());*/

    image::Rgba32FImage::from_raw(output_dim.width, output_dim.height, positions_floats)
        .unwrap()
        .save("positions.exr")
        .unwrap();
    image::Rgba32FImage::from_raw(output_dim.width, output_dim.height, normals_floats)
        .unwrap()
        .save("normals.exr")
        .unwrap();
    */

    Ok(())
}

fn slice_to_bytes(
    buffer_slice: &wgpu::BufferSlice,
    device: &wgpu::Device,
    extent: wgpu::Extent3d,
    padded_width: u32,
) -> Vec<f32> {
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::Wait);

    if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
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

pub struct NodeTree {
    inner: Vec<(glam::Mat4, usize)>,
}

impl NodeTree {
    pub fn new(gltf: &goth_gltf::Gltf<goth_gltf::default_extensions::Extensions>) -> Self {
        let mut inner = vec![(glam::Mat4::IDENTITY, usize::max_value()); gltf.nodes.len()];

        for (index, node) in gltf.nodes.iter().enumerate() {
            inner[index].0 = match node.transform() {
                goth_gltf::NodeTransform::Matrix(matrix) => glam::Mat4::from_cols_array(&matrix),
                goth_gltf::NodeTransform::Set {
                    translation,
                    rotation,
                    scale,
                } => glam::Mat4::from_scale_rotation_translation(
                    scale.into(),
                    glam::Quat::from_array(rotation),
                    translation.into(),
                ),
            };
            for child in &node.children {
                inner[*child].1 = index;
            }
        }

        Self { inner }
    }

    pub fn transform_of(&self, mut index: usize) -> glam::Mat4 {
        let mut transform_sum = glam::Mat4::IDENTITY;

        while index != usize::max_value() {
            let (transform, parent_index) = self.inner[index];
            transform_sum = transform * transform_sum;
            index = parent_index;
        }

        transform_sum
    }
}

fn collect_buffer_view_map(
    gltf: &goth_gltf::Gltf<Extensions>,
    glb_buffer: Option<&[u8]>,
    base_path: &Path,
) -> anyhow::Result<HashMap<usize, Vec<u8>>> {
    use std::borrow::Cow;

    let mut buffer_map = HashMap::new();

    if let Some(glb_buffer) = glb_buffer {
        buffer_map.insert(0, Cow::Borrowed(glb_buffer));
    }

    for (index, buffer) in gltf.buffers.iter().enumerate() {
        let uri = match &buffer.uri {
            Some(uri) => uri,
            None => continue,
        };

        if &uri[..4] == "data" {
            panic!("{:?}", uri)
            /*
            let (_mime_type, data) = uri
                .path()
                .split_once(',')
                .ok_or_else(|| anyhow::anyhow!("Failed to get data uri split"))?;

            log::warn!("Loading buffers from embedded base64 is inefficient. Consider moving the buffers into a seperate file.");
            buffer_map.insert(index, Cow::Owned(base64::decode(data)?));
            */
        } else {
            let path = base_path.parent().unwrap().join(&uri);

            buffer_map.insert(index, Cow::Owned(std::fs::read(&path).unwrap()));
        }
    }

    let mut buffer_view_map = HashMap::new();

    for (i, buffer_view) in gltf.buffer_views.iter().enumerate() {
        if let Some(buffer) = buffer_map.get(&buffer_view.buffer) {
            buffer_view_map.insert(
                i,
                buffer[buffer_view.byte_offset..buffer_view.byte_offset + buffer_view.byte_length]
                    .to_vec(),
            );
        }
    }

    Ok(buffer_view_map)
}
