+++ title = "Collatz@Home: a distributed computing system powered by WebGPU" date = 2025-12-20T08:00:00+00:00 [params] math = true +++

WebGPU is a cross platform graphics API supported by most major browsers and operating systems. It was started in 2021 and has keen interest from major tech companies such as Microsoft and Apple (other tech companies are available). Perhaps most notable about WebGPU is that it is able to run in-browser with efficiency comparable to native GPU applications.

There are huge possibilities for this technology, and as a Christmas treat I thought I'd write a tutorial that I havent seen elsewhere. [Folding@home](https://foldingathome.org) is a project that lets individuals 'donate' their compute to science and medicine. The idea is that you may download the program and run it, letting your computer process parts of protein folding simulations that aids in scientific research and contributes to medicine. My idea is to create a similar distributed computing application that needs only an open browser tab to contribute. Protein folding is a huge challenge that I don't mean to take on myself so instead I picked a nice simple one that suits the style of my blog.

The Collatz conjecture is as follows:
$$
T(n)=
\begin{cases}
\frac{n}{2}, & n\equiv 0 \pmod 2,\\[6pt]
3n+1, & n\equiv 1 \pmod 2.
\end{cases}
$$
Recursively call this function on a number, for all numbers (n), if you call the function enough times it will reach 1. This conjecture has not been proved, but every number we have ever tried did end up at 1 in the end. It's not certain how many numbers have been attempted but [this paper published May 2025 in the 'Journal of Supercomputing'](https://link.springer.com/article/10.1007/s11227-025-07337-0?utm_source=chatgpt.com) says the conjecture holds at least up to $2^{71}$

While its not likely that we will disprove the conjecture it's worth a shot and is a good substitute for protein folding for the purposes of what we are trying to learn.

# Computation on the CPU

This is what the function looks like in Rust. Note that I have chosen to use unsigned 128 bit integers (`u128`) because we're aiming to compute numbers higher than the previous limit of $2^{71}$

```rust
fn collatz(mut n: u128) -> (u128, u128) {
    // count the number of steps we've done and highest number reached
    let mut steps: u128 = 0;
    let mut max = n;
    // loop until we reach 1
    while n != 1 {
        if n % 2 == 0 {
            // if n is even divide by 2
            n = n / 2;
        } else {
            // if n is odd do 3n + 1
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

This function is only partially correct. If the conjecture is false we could enter a never ending cycle without reaching 1. We can't just keep a list of seen values to check for a cycle as this would be spacially inneficient (we also dont have hash sets in the GPU). However we can use the 'Tortoise and Hare algorithm' where we have two variables:
1. The Hare who moves every iteration
2. The Tortoise who moves every other iteration

We can provably state that if the variables ever meet (have the same value) in an iteration then there must be a cycle, and if there is a cycle then the variables will meet.

```rust
fn collatz_step(n: u128) -> u128 {
    if n % 2 == 0 {
        n / 2
    } else {
        3 * n + 1
    }
}

fn collatz(mut n: u128) -> Result<(u128, u128), &'static str> {
    let mut steps: u128 = 0;
    let mut max = n;

    // Tortoise–hare setup
    let mut tortoise = n;
    let mut hare = collatz_step(n); // hare starts one step ahead or they have already met

    while hare != 1 {
        // hare moves every step
        hare = collatz_step(hare);

        // tortoise moves every other step
        if steps % 2 == 0 {
            tortoise = collatz_step(tortoise);
        }

        if hare > max {
            max = hare;
        }
        steps += 1;

        if hare == tortoise {
            return Err("Cycle detected in Collatz sequence");
        }
    }

    Ok((steps, max))
}
```

There are also possible issues with overflows but I'll leave you to deal with these ;).

# GPU is tricky
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