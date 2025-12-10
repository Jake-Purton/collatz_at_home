steps: create the tester function for collatz

```
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
```

create easy wgsl function and test TODO test that this works and write appropriate code

```wgsl
struct CollatzResult {
    steps: u32,
    max: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<CollatzResult>;

fn collatz_u32(n_input: u32) -> CollatzResult {
    var n = n_input;
    var steps: u32 = 0u;
    var max_val: u32 = n;

    loop {
        if (n == 1u) {
            break;
        }

        // Safety limit to avoid infinite GPU loops
        if (steps >= 100000u) {
            break;
        }

        if ((n & 1u) == 0u) {
            // Even: n = n / 2
            n = n >> 1u;
        } else {
            // Odd: n = 3*n + 1
            n = 3u * n + 1u;
        }

        if (n > max_val) {
            max_val = n;
        }

        steps++;
    }

    var res: CollatzResult;
    res.steps = steps;
    res.max = max_val;
    return res;
}

// distributes all of the workers with an id (an index in the array of inputs)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx < arrayLength(&input)) {
        // write out the result of the index you read in
        output[idx] = collatz_u32(input[idx]);
    }
}
```

`@group(0) @binding(0) var<storage, read> input: array<u32>;`
there will be an input which is read only and contains an array of u32s

`@group(0) @binding(1) var<storage, read_write> output: array<CollatzResult>;`
you also have an output which is an array of results, and is read-write

create U128 wgsl

add overflow checking

compile for web

create a webserver