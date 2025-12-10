mod debug;

use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

// 50,000 is 1mb
const RANGE: u128 = 50_000;

// Helper function to convert u128 to array of 4 u32s (little-endian)
fn u128_to_u32_array(n: u128) -> [u32; 4] {
    [
        (n & 0xFFFFFFFF) as u32,
        ((n >> 32) & 0xFFFFFFFF) as u32,
        ((n >> 64) & 0xFFFFFFFF) as u32,
        ((n >> 96) & 0xFFFFFFFF) as u32,
    ]
}

// Helper function to convert array of 4 u32s back to u128 (little-endian)
fn u32_array_to_u128(parts: &[u32; 4]) -> u128 {
    (parts[0] as u128)
        | ((parts[1] as u128) << 32)
        | ((parts[2] as u128) << 64)
        | ((parts[3] as u128) << 96)
}

// Helper function to convert u32 array to bytes
fn u32_array_to_bytes(parts: &[u32; 4]) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    for (i, &part) in parts.iter().enumerate() {
        let part_bytes = part.to_le_bytes();
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&part_bytes);
    }
    bytes
}

#[wasm_bindgen]
pub async fn check_webgpu_support() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await;

    adapter.is_ok()
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log!("WASM module initialized!");
}

#[wasm_bindgen]
pub async fn do_gpu_collatz(start_n: String) -> Result<Vec<u32>, JsValue> {
    console_log!("hello here");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });
    console_log!("made it here 0");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await;

    let adapter = match adapter {
        Ok(a) => {
            console_log!("Adapter found: {:?}", a.get_info().name);
            a
        }
        Err(e) => {
            console_log!(
                "ERROR: No GPU adapter found. WebGPU may not be supported in this browser. {:?}",
                e
            );
            return Err(JsValue::from_str(
                "No GPU adapter found. Try Chrome WebGPU enabled. Safari Does not support WebGPU",
            ));
        }
    };
    console_log!("made it here 1");

    let (device, queue) = match adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
    {
        Ok(a) => a,
        Err(e) => {
            console_log!("{e}");
            return Err(JsValue::from_str(&format!("{e}")));
        }
    };

    console_log!("made it here 2");

    // parse start n
    let n = if let Ok(n) = start_n.parse::<u128>() {
        n
    } else {
        return Err(JsValue::from_str("Could not parse n"));
    };

    let test_numbers: Vec<u128> = (n..n + RANGE).collect();

    // Convert to GPU format (4 × u32 per number)
    let input_data: Vec<u8> = test_numbers
        .iter()
        .flat_map(|&n| u32_array_to_bytes(&u128_to_u32_array(n)))
        .collect();

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: &input_data,
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Output: Each result has steps (u32=4 bytes) + max (4×u32=16 bytes) = 20 bytes, but align to 32 bytes
    let output_size = test_numbers.len() * 32; // Struct padding for alignment

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Collatz Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("add.wgsl").into()),
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
        label: Some("Bind Group"),
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // Dispatch enough workgroups to cover all input numbers
        let workgroup_size = 64;
        let num_workgroups = (test_numbers.len() as u32 + workgroup_size - 1) / workgroup_size;
        cpass.dispatch_workgroups(num_workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size as u64);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);

    // In WASM, we need to use a channel to properly await the buffer mapping
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    // Poll the device until the buffer is mapped
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    // Wait for the mapping to complete
    receiver
        .recv_async()
        .await
        .map_err(|e| JsValue::from_str(&format!("Channel error: {}", e)))?
        .map_err(|e| JsValue::from_str(&format!("Buffer mapping failed: {:?}", e)))?;

    let data = buffer_slice.get_mapped_range();
    let results: &[u32] = bytemuck::cast_slice(&data);

    for (i, &n) in test_numbers.iter().enumerate() {
        let offset = i * 5;
        let steps = results[offset];
        let max_parts = [
            results[offset + 1],
            results[offset + 2],
            results[offset + 3],
            results[offset + 4],
        ];
        let max_value = u32_array_to_u128(&max_parts);

        if n % 25_000 == 0 {
            console_log!("n: {n}, steps: {steps}, max_value: {max_value}")
        }
    }

    let vec_results = results.to_vec();

    drop(data);
    staging_buffer.unmap();

    Ok(vec_results)
}
