
#pragma once
#ifndef RVV_MATMUL_H
#define RVV_MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif

struct exo_win_1f32{
    float *data;
    int_fast32_t strides[1];
};
typedef struct rvv_matmul_Context { 

} rvv_matmul_Context;


// rank_k_reduce_6x16_scheduled(
//     K : size,
//     C : f32[6,16]  @DRAM,
//     A : f32[6,K]  @DRAM,
//     B : f32[K,16]  @DRAM
// )
void rank_k_reduce_6x16_scheduled( rvv_matmul_Context *ctxt, int_fast32_t K, float* C, float* A, float* B );

// rank_k_reduce_6x16(
//     K : size,
//     C : f32[6,16]  @DRAM,
//     A : f32[6,K]  @DRAM,
//     B : f32[K,16]  @DRAM
// )
void rank_k_reduce_6x16( rvv_matmul_Context *ctxt, int_fast32_t K, float* C, float* A, float* B );



#ifdef __cplusplus
}
#endif
#endif  // RVV_MATMUL_H
