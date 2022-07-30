from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *


class rvv32f(Memory):
    pass


# V extension

# vlm.v vd, (rs1)   #  Load byte vector of length ceil(vl/8)
# VLEN=4 * 8b

@instr('vlm.v({dst_data}, &{scr_data});')
def vlm_v(
        dst: [8][4] @ rvv32f,
        src: [8][4] @ DRAM
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 4):
        dst[i] = src[i]


# vsm.v vs3, (rs1)  #  Store byte vector of length ceil(vl/8)
@instr('vsm.v(&{dst_data}, {src_data});')
def vsm_v(
        dst: [8][4] @ DRAM,
        src: [8][4] @ rvv32f
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 4):
        dst[i] = src[i]


# vfmul.vv vd, vs2, vs1, vm Vector-vector
@instr('vfmul.vv({dst}, {src1}, {src2});')
def vfmul_v(
        dst: [8][4] @ rvv32f,
        src1: [8][4] @ rvv32f,
        src2: [8][4] @ rvv32f,
):
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1

    for i in range(0, 4):
        dst[i] = src1[i] * src2[i]


# vredsum.vs  vd, vs2, vs1, vm : vd[0] =  sum( vs1[0] , vs2[*] )
@instr('vredsum.vs({dst}, {src1}, {src2});')
def vredsum_vs(
        dst: [8][4] @ rvv32f,
        src1: [8][4] @ rvv32f,
        src2: [8][4] @ rvv32f
):
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1

    dst[0] = src1[0]
    for i in range(0, 4):
        dst[0] += src2[i]


# vlse32.v vd, (rs1), rs2, vm 32-bit strided load
@instr('vlse32.v({dst}, &{src1}, {src2});')
def vlse32_v(
        dst: [8][4] @ rvv32f,
        src1: [8][256] @ DRAM,
        src2: [8][4] @ rvv32f
):
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1

    for i in par(0, 4):
        dst[i] = src1[src2[i] * i]


