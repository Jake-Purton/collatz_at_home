/* tslint:disable */
/* eslint-disable */

export function do_gpu_collatz(start_n: string): Promise<void>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly do_gpu_collatz: (a: number, b: number) => any;
  readonly wasm_bindgen__convert__closures_____invoke__h51a270327fca4ed5: (a: number, b: number, c: any) => void;
  readonly wasm_bindgen__closure__destroy__h6a0f5aa50c23d88e: (a: number, b: number) => void;
  readonly wasm_bindgen__convert__closures_____invoke__hdf270ce0da308ff1: (a: number, b: number, c: any) => void;
  readonly wasm_bindgen__closure__destroy__hd7b7e163837b9c9e: (a: number, b: number) => void;
  readonly wasm_bindgen__convert__closures_____invoke__h089a09d160a6520b: (a: number, b: number, c: any, d: any) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
