use goth_gltf::default_extensions::Extensions;
use std::collections::HashMap;
use std::path::Path;

pub fn init_wgpu() -> anyhow::Result<(wgpu::Device, wgpu::Queue)> {
    let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    let instance = wgpu::Instance::new(backend);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("No suitable GPU adapters found on the system!");

    Ok(pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("device"),
            features: adapter.features(),
            limits: adapter.limits(),
        },
        None,
    ))?)
}

pub fn slice_to_bytes(
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

pub fn collect_buffer_view_map(
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
