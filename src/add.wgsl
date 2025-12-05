// collatz.wgsl - Multi-precision u128 implementation
struct U128 {
    parts: array<u32, 4>  // Little-endian: [low, mid_low, mid_high, high]
}

struct CollatzResult {
    steps: u32,
    max: U128
}

@group(0) @binding(0) var<storage, read> input: array<U128>;
@group(0) @binding(1) var<storage, read_write> output: array<CollatzResult>;

// Check if U128 equals 1
fn is_one(n: U128) -> bool {
    return n.parts[0] == 1u && n.parts[1] == 0u && n.parts[2] == 0u && n.parts[3] == 0u;
}

// Check if U128 is even (LSB is 0)
fn is_even(n: U128) -> bool {
    return (n.parts[0] & 1u) == 0u;
}

// Divide U128 by 2 (right shift by 1)
fn div_by_2(n: U128) -> U128 {
    var result: U128;
    result.parts[0] = (n.parts[0] >> 1u) | ((n.parts[1] & 1u) << 31u);
    result.parts[1] = (n.parts[1] >> 1u) | ((n.parts[2] & 1u) << 31u);
    result.parts[2] = (n.parts[2] >> 1u) | ((n.parts[3] & 1u) << 31u);
    result.parts[3] = n.parts[3] >> 1u;
    return result;
}

// Add two U128 numbers
fn add_u128(a: U128, b: U128) -> U128 {
    var result: U128;
    var carry = 0u;
    
    // Add part 0
    let sum0 = a.parts[0] + b.parts[0];
    result.parts[0] = sum0;
    carry = u32(sum0 < a.parts[0]);
    
    // Add part 1
    let sum1 = a.parts[1] + b.parts[1] + carry;
    result.parts[1] = sum1;
    carry = u32(sum1 < a.parts[1] || (carry == 1u && sum1 == a.parts[1]));
    
    // Add part 2
    let sum2 = a.parts[2] + b.parts[2] + carry;
    result.parts[2] = sum2;
    carry = u32(sum2 < a.parts[2] || (carry == 1u && sum2 == a.parts[2]));
    
    // Add part 3
    let sum3 = a.parts[3] + b.parts[3] + carry;
    result.parts[3] = sum3;
    
    return result;
}

// Multiply U128 by 3 and add 1: (n * 3) + 1 = n + n + n + 1
fn mul_3_add_1(n: U128) -> U128 {
    let doubled = add_u128(n, n);
    let tripled = add_u128(doubled, n);
    
    // Add 1
    var result = tripled;
    result.parts[0] = result.parts[0] + 1u;
    
    // Handle carry propagation from adding 1
    if (result.parts[0] == 0u) {
        result.parts[1] = result.parts[1] + 1u;
        if (result.parts[1] == 0u) {
            result.parts[2] = result.parts[2] + 1u;
            if (result.parts[2] == 0u) {
                result.parts[3] = result.parts[3] + 1u;
            }
        }
    }
    
    return result;
}

// Compare two U128 values: returns true if a > b
fn greater_than(a: U128, b: U128) -> bool {
    if (a.parts[3] != b.parts[3]) { return a.parts[3] > b.parts[3]; }
    if (a.parts[2] != b.parts[2]) { return a.parts[2] > b.parts[2]; }
    if (a.parts[1] != b.parts[1]) { return a.parts[1] > b.parts[1]; }
    return a.parts[0] > b.parts[0];
}

// Collatz computation
fn collatz(n_input: U128) -> CollatzResult {
    var n = n_input;
    var steps = 0u;
    var max = n;
    
    // Limit iterations to prevent infinite loops
    for (var iter = 0u; iter < 10000u; iter++) {
        if (is_one(n)) {
            break;
        }
        
        if (is_even(n)) {
            n = div_by_2(n);
        } else {
            n = mul_3_add_1(n);
        }
        
        if (greater_than(n, max)) {
            max = n;
        }
        
        steps++;
    }
    
    var result: CollatzResult;
    result.steps = steps;
    result.max = max;
    return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < arrayLength(&input)) {
        output[idx] = collatz(input[idx]);
    }
}
