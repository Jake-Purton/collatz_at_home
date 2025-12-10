cargo build --target wasm32-unknown-unknown --release
~/.cargo/bin/wasm-bindgen --target web --out-dir dist target/wasm32-unknown-unknown/release/collatz_at_home.wasm
cp ./index.html ./dist/
cd dist/
python3 -m http.server 8000