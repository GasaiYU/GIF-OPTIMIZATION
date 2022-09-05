from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *

import os
from pathlib import Path
import sys

from exo.memory import Memory, DRAM, MemGenError


# Need fixing?
# Compared to AVX512, there's no need to restrict vector's width
class rvv32f(Memory):
    @classmethod
    def global_(cls):
        return "#include <riscv_vector.h>"

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(f'{srcinfo}: RVV vectors are not scalar values')
        if not prim_type == "float":
            raise MemGenError(f'{srcinfo}: RVV vectors must be f32 (for now)')
        shape = shape[:-1]
        if shape:
            result = f'vfloat32m8_t {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f'vfloat32m8_t {new_name};'
        return result

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ''

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        assert strides[-1] == '1'
        idxs = indices[:-1] or ''
        if idxs:
            idxs = '[' + ']['.join(idxs) + ']'
        return f'{baseptr}{idxs}'


# V extension

# vlm.v vd, (rs1)   #  Load byte vector of length ceil(vl/8)
# VLEN=4 * 8b

@instr('{dst_data} = vlm.v(&{src_data});')
def vlm_v(
        dst: [f32][8] @ rvv32f,
        src: [f32][8] @ DRAM
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] = src[i]


# vsm.v vs3, (rs1)  #  Store byte vector of length ceil(vl/8)
@instr('vsm.v(&{dst_data}, {src_data});')
def vsm_v(
        dst: [f32][8] @ DRAM,
        src: [f32][8] @ rvv32f
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] = src[i]


@instr('vlse32.v(&{dst_data}, {src1}, #0, #0xffffffff;')
def vlse32_v(
        dst: [f32][8] @ rvv32f,
        src1: [f32][8] @ DRAM,
):
    assert stride(src1, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] = src1[0]


# vfmul.vv vd, vs2, vs1, vm Vector-vector
@instr('vfmul.vv({dst}, {src1}, {src2});')
def vfmul_v(
        dst: [f32][8] @ rvv32f,
        src1: [f32][8] @ rvv32f,
        src2: [f32][8] @ rvv32f,
):
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1

    for i in par(0, 8):
        dst[i] = src1[i] * src2[i]


# vredsum.vs  vd, vs2, vs1, vm : vd[0] =  sum( vs1[0] , vs2[*] )
@instr('vredsum.vs({dst}, {src1}, {src2});')
def vredsum_vs(
        dst: [f32][8] @ rvv32f,
        src1: [f32][8] @ rvv32f,
        src2: [f32][8] @ rvv32f
):
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1

    dst[0] = src1[0]
    for i in par(0, 8):
        dst[0] += src2[i]


@instr('{dst}, {src1} ,{src2}')
def vfmacc_vv(
    dst: [f32][8] @ rvv32f,
    src1: [f32][8] @ rvv32f,
    src2: [f32][8] @ rvv32f
):
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1

    for i in par(0, 8):
        dst[i] += src1[i] * src2[i]


"""
# vlse32.v vd, (rs1), rs2, vm 32-bit strided load
@instr('vlse32.v({dst}, &{src1}, {src2});')
def vlse32_v(
        dst: [f32][8] @ AVX2,
        src1: [f32][8] @ DRAM,
        src2: [f32][8] @ AVX2
):
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1

    for i in par(0, 8):
        dst[i] = src1[src2[i] * i]
"""

# Hide output when running through exocc.
if __name__ != '__main__' and hasattr(os, 'devnull'):
    sys.stdout = open(os.devnull, 'w')


# Algorithm Definition
@proc
def rank_k_reduce_6x16(
        K: size,
        C: f32[6, 16] @ DRAM,
        A: f32[6, K] @ DRAM,
        B: f32[K, 16] @ DRAM,
):
    for i in seq(0, 6):
        for j in seq(0, 16):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


print("Original algorithm:")
print(rank_k_reduce_6x16)

# First Block

rvv = rank_k_reduce_6x16.rename("rank_k_reduce_6x16_scheduled")
rvv = rvv.stage_assn('C_reg', 'C[_] += _')
rvv = rvv.set_memory('C_reg', rvv32f)
print("First Block")
print(rvv)

# Second block

rvv = rvv.split('j', 8, ['jo', 'ji'], perfect=True)
rvv = rvv.reorder('ji', 'k')
rvv = rvv.reorder('jo', 'k')
rvv = rvv.reorder('i', 'k')
print("Second block:")
print(rvv)

# Third block

rvv = rvv.lift_alloc('C_reg:_', n_lifts=3)
rvv = rvv.fission_after('C_reg = _ #0', n_lifts=3)
rvv = rvv.fission_after('C_reg[_] += _ #0', n_lifts=3)
rvv = rvv.lift_alloc('C_reg:_', n_lifts=1)
rvv = rvv.fission_after('for i in _:_#0', n_lifts=1)
rvv = rvv.fission_after('for i in _:_#1', n_lifts=1)
rvv = rvv.simplify()
print("Third block:")
print(rvv)

# Fourth block

rvv = rvv.bind_expr('a_vec', 'A[i, k]')
rvv = rvv.set_memory('a_vec', rvv32f)
rvv = rvv.lift_alloc('a_vec:_', keep_dims=True)
rvv = rvv.fission_after('a_vec[_] = _')
print("Fourth block:")
print(rvv)

# Fifth block

rvv = rvv.bind_expr('b_vec', 'B[k, _]')
rvv = rvv.set_memory('b_vec', rvv32f)
rvv = rvv.lift_alloc('b_vec:_', keep_dims=True)
rvv = rvv.fission_after('b_vec[_] = _')
print("Fifth block:")
print(rvv)

# Sixth block

# rvv = rvv.replace_all(vlm_v)
# rvv = rvv.replace_all(vsm_v)
rvv = rvv.replace_all(vfmacc_vv)
rvv = rvv.replace_all(vlse32_v)
rvv = rvv.replace(vlm_v, 'for ji in _:_ #0')
rvv = rvv.replace(vlm_v, 'for ji in _:_ #0')
rvv = rvv.replace(vsm_v, 'for ji in _:_ #0')
print("Sixth block:")
print(rvv)
