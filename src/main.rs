use wgpu::util::DeviceExt;
use std::fs::File;
use std::io::Write;

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

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();

    // Test with some interesting Collatz numbers
    let test_numbers: Vec<u128> = ((1_u128 << 100)..(1_u128 << 100)+1000000).collect();

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
        entry_point: "main",
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
        ],
        label: Some("Bind Group"),
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compute Pass")});
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
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = buffer_slice.get_mapped_range();
    let results: &[u32] = bytemuck::cast_slice(&data);
    
    // Parse results: 5 u32s per result (steps + 4 for U128)
    let mut output = String::new();
    output.push_str("Collatz Results:\n");
    
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
        
        let line = format!("n={}: steps={}, max={}\n", n, steps, max_value);
        output.push_str(&line);
        // print!("  {}", line);
    }
    
    // Write to file
    let mut file = File::create("collatz_results.txt").expect("Failed to create file");
    file.write_all(output.as_bytes()).expect("Failed to write to file");
    println!("\nResults written to collatz_results.txt");
    
    drop(data);
    staging_buffer.unmap();
}

fn collatz(mut n: u128) -> (u128, u128) {
    let mut steps: u128 = 0;
    let mut max = n;
    while n != 1 {
        if n % 2 == 0 {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
        if n > max {
            max = n;
        }
        steps += 1;
    }
    return (steps, max);
}
