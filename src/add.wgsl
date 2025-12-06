struct U128 {
    parts: array<u32, 4>  // Little-endian: [low, mid_low, mid_high, high]
}

struct CollatzResult {
    steps: u32,
    max: U128
}

@group(0) @binding(0) var<storage, read> input: array<U128>;
@group(0) @binding(1) var<storage, read_write> output: array<CollatzResult>;

fn is_one(n: U128) -> bool {
    return n.parts[0] == 1u && n.parts[1] == 0u && n.parts[2] == 0u && n.parts[3] == 0u;
}

fn is_even(n: U128) -> bool {
    return (n.parts[0] & 1u) == 0u;
}

fn div_by_2(n: U128) -> U128 {
    var result: U128;
    // right shift by 1 and get the bottom of the highrt one
    result.parts[0] = (n.parts[0] >> 1u) | ((n.parts[1] & 1u) << 31u);
    result.parts[1] = (n.parts[1] >> 1u) | ((n.parts[2] & 1u) << 31u);
    result.parts[2] = (n.parts[2] >> 1u) | ((n.parts[3] & 1u) << 31u);
    result.parts[3] = n.parts[3] >> 1u;
    return result;
}

struct U128AddResult {
    value: U128,
    carry: u32,      // 1 if overflowed past 128 bits
};

fn add_u128(a: U128, b: U128) -> U128AddResult {
    var total: U128;
    var res: U128AddResult;
    var carry = 0u;

    let sum0 = a.parts[0] + b.parts[0];
    total.parts[0] = sum0;
    carry = u32(sum0 < a.parts[0]);

    let sum1 = a.parts[1] + b.parts[1] + carry;
    total.parts[1] = sum1;
    carry = u32(sum1 < a.parts[1] || (carry == 1u && sum1 == a.parts[1]));

    let sum2 = a.parts[2] + b.parts[2] + carry;
    total.parts[2] = sum2;
    carry = u32(sum2 < a.parts[2] || (carry == 1u && sum2 == a.parts[2]));

    let sum3 = a.parts[3] + b.parts[3] + carry;
    total.parts[3] = sum3;
    carry = u32(sum3 < a.parts[3] || (carry == 1u && sum3 == a.parts[3]));

    res.value=total;
    res.carry=carry;
    return res;
}


fn mul_3_add_1(n: U128) -> U128 {
    let doubled = add_u128(n, n);
    let tripled = add_u128(doubled, n);
    
    var result = tripled;
    result.parts[0] = result.parts[0] + 1u;
    
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

fn greater_than(a: U128, b: U128) -> bool {
    if (a.parts[3] != b.parts[3]) { return a.parts[3] > b.parts[3]; }
    if (a.parts[2] != b.parts[2]) { return a.parts[2] > b.parts[2]; }
    if (a.parts[1] != b.parts[1]) { return a.parts[1] > b.parts[1]; }
    return a.parts[0] > b.parts[0];
}

// Check if two U128 values are equal
fn equals(a: U128, b: U128) -> bool {
    return a.parts[0] == b.parts[0] && a.parts[1] == b.parts[1] && 
           a.parts[2] == b.parts[2] && a.parts[3] == b.parts[3];
}

fn collatz(n_input: U128) -> CollatzResult {
    var n = n_input;
    var steps = 0u;
    var max = n;
    
    // Floyd's cycle detection: slow and fast pointers
    var slow = n;
    var fast = n;
    var slow_steps = 0u;
    
    loop {
        if (is_one(n)) {
            break;
        }
        
        // Safety limit to prevent GPU hangs
        if (steps >= 100000u) {
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
        
        // Cycle detection: advance slow pointer every other step
        if (steps % 2u == 0u) {
            if (is_even(slow)) {
                slow = div_by_2(slow);
            } else {
                slow = mul_3_add_1(slow);
            }
            slow_steps++;
            
            // Check if we've found a cycle (fast caught up to slow)
            if (equals(n, slow) && steps > 2u) {
                // Cycle detected, break out
                break;
            }
        }
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
