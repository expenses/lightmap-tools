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
