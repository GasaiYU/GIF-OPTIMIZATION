#include "rvv_matmul.h"

static int _floor_div(int num, int quot) {
  int off = (num>=0)? 0 : quot-1;
  return (num-off)/quot;
}

static int8_t _clamp_32to8(int32_t x) {
  return (x < -128)? -128 : ((x > 127)? 127 : x);
}

#include <stdio.h>
#include <stdlib.h>


#include <riscv_vector.h>


/* relying on the following instruction...
vlse32_v(dst,src1)
vlse32.v(&{dst_data}, {src1}, #0, #0xffffffff;
*/


/* relying on the following instruction...
vsm_v(dst,src)
vsm.v(&{dst_data}, {src_data});
*/


/* relying on the following instruction...
vfmacc_vv(dst,src1,src2)
{dst}, {src1} ,{src2}
*/


/* relying on the following instruction...
vlm_v(dst,src)
{dst_data} = vlm.v(&{src_data});
*/

// rank_k_reduce_6x16_scheduled(
//     K : size,
//     C : f32[6,16]  @DRAM,
//     A : f32[6,K]  @DRAM,
//     B : f32[K,16]  @DRAM
// )
void rank_k_reduce_6x16_scheduled( rvv_matmul_Context *ctxt, int_fast32_t K, float* C, float* A, float* B ) {
vfloat32m8_t C_reg[(K + 1)][6][2];
for (int k = 0; k < K; k++) {
  for (int i = 0; i < 6; i++) {
    for (int jo = 0; jo < 2; jo++) {
      C_reg[k + 0][i + 0][jo + 0] = vlm.v(&C[(i + 0) * (16) + (8 * jo + 0) * (1)]);
    }
  }
}
for (int k = 0; k < K; k++) {
  for (int i = 0; i < 6; i++) {
    for (int jo = 0; jo < 2; jo++) {
      vfloat32m8_t a_vec;
      vlse32.v(&a_vec, ((struct exo_win_1f32){ (float*)&A[(i + 0) * (K) + (k + 0) * (1)], { 1 } }), #0, #0xffffffff;
      vfloat32m8_t b_vec;
      b_vec = vlm.v(&B[(k + 0) * (16) + (8 * jo + 0) * (1)]);
      ((struct exo_win_1f32){ (float*)&C_reg[k + 0][i + 0][jo + 0], { 1 } }), ((struct exo_win_1f32){ (float*)&a_vec, { 1 } }) ,((struct exo_win_1f32){ (float*)&b_vec, { 1 } })
    }
  }
}
for (int k = 0; k < K; k++) {
  for (int i = 0; i < 6; i++) {
    for (int jo = 0; jo < 2; jo++) {
      vsm.v(&C[(i + 0) * (16) + (8 * jo + 0) * (1)], C_reg[k + 0][i + 0][jo + 0]);
    }
  }
}
}

// rank_k_reduce_6x16(
//     K : size,
//     C : f32[6,16]  @DRAM,
//     A : f32[6,K]  @DRAM,
//     B : f32[K,16]  @DRAM
// )
void rank_k_reduce_6x16( rvv_matmul_Context *ctxt, int_fast32_t K, float* C, float* A, float* B ) {
for (int i = 0; i < 6; i++) {
  for (int j = 0; j < 16; j++) {
    for (int k = 0; k < K; k++) {
      C[(i) * (16) + (j) * (1)] += A[(i) * (K) + (k) * (1)] * B[(k) * (16) + (j) * (1)];
    }
  }
}
}
