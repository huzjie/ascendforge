# TileLang example: tiled GEMM
program gemm_kernel(128, 128, 32) {
    tile A : fp16[128, 32] @ GM
    tile B : fp16[32, 128] @ GM
    tile C : fp16[128, 128] @ L1
    load(in=A, out=A)
    load(in=B, out=B)
    gemm(in=A,B, out=C)
    store(in=C, out=C)
}
