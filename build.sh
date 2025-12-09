cargo build --target wasm32-unknown-unknown --release
wasm-bindgen --target web --out-dir dist target/wasm32-unknown-unknown/release/collatz_at_home.wasm
cp ./index.html ./dist/

npx serve dist