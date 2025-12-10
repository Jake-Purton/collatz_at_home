# **Collatz@Home**

Collatz@Home is an experimental **WebGPU-accelerated Collatz sequence explorer** written in **Rust → WebAssembly + WGSL**.
The goal is to offload millions of independent Collatz computations to the user’s GPU directly inside the browser—similar in spirit to “distributed computing”, but powered entirely by WebGPU. This could help mathematicians spot patterns to help possibly solve the conjecture.

---

## **Tech Overview**

### **Rust (WASM)**

* Initializes WebGPU.
* Allocates buffers.
* Dispatches compute workgroups.
* Maps output buffers back to WASM memory.
* Logs important results.

### **WGSL Shader**

Implements:

* 128-bit arithmetic (`add_u128`, `mul_3_add_1`, division by 2) and overflow checking
* Collatz iteration with cycle detection
* Per-index output structs written to GPU storage buffers

Each shader invocation handles **one number**, making the algorithm embarrassingly parallel.

---

## **How to Run**

1. Use a browser with **WebGPU enabled**:

   * Chrome 113+
   * Edge 113+
   * Safari but you need to enable WebGPU

2. Build the WASM into /dist and serve:

    feel free to use any http server. I have used `npx serve dist`
    ```
    ./build.sh
    ```
3. Open in your browser
---
