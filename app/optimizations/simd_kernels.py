"""
SIMD kernels optimized for AMD EPYC 7R13 processor.
Implements AVX2 assembly optimizations for matrix and vector operations.
"""

import ctypes
import numpy as np
from typing import Tuple, Optional
from loguru import logger
import platform
import os


class SIMDKernels:
    """SIMD optimized kernels using AVX2 for AMD EPYC 7R13."""
    
    def __init__(self):
        self.lib = None
        self.avx2_supported = self._check_avx2_support()
        self.epyc_optimized = self._check_epyc_processor()
        
        if self.avx2_supported:
            self._compile_and_load_kernels()
        else:
            logger.warning("AVX2 not supported on this processor, falling back to standard operations")
    
    def _check_avx2_support(self) -> bool:
        """Check if AVX2 is supported on the current processor."""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            return 'avx2' in flags and 'fma' in flags
        except ImportError:
            # Fallback method using /proc/cpuinfo on Linux
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    return 'avx2' in content and 'fma' in content
            except:
                logger.warning("Cannot determine AVX2 support, assuming not available")
                return False
    
    def _check_epyc_processor(self) -> bool:
        """Check if running on AMD EPYC processor."""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            brand = info.get('brand_raw', '').lower()
            return 'epyc' in brand or 'amd' in brand
        except ImportError:
            # Fallback method
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read().lower()
                    return 'epyc' in content or 'amd' in content
            except:
                return False
    
    def _compile_and_load_kernels(self):
        """Compile and load the optimized C library with inline assembly."""
        try:
            # Create the C source file with inline assembly
            self._create_c_source()
            
            # Compile with optimizations for AMD EPYC
            self._compile_library()
            
            # Load the compiled library
            self._load_library()
            
            logger.info("Successfully compiled and loaded AVX2 optimized kernels")
        except Exception as e:
            logger.error(f"Failed to compile/load optimized kernels: {e}")
            self.lib = None
    
    def _create_c_source(self):
        """Create the C source file with AVX2 assembly optimizations."""
        c_source = '''
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Optimized matrix multiplication kernel for AMD EPYC 7R13
// Uses micro-kernels with register blocking and prefetching
void gemm_avx2_epyc(const float* A, const float* B, float* C, 
                    int M, int N, int K) {
    
    // AMD EPYC optimized tile sizes based on cache hierarchy
    // L1: 32KB, L2: 512KB, L3: 256MB (shared across CCX)
    const int TILE_M = 96;   // Optimized for register pressure
    const int TILE_N = 96;   
    const int TILE_K = 512;  // L2 cache friendly
    const int MC = 192;      // L3 cache blocking
    const int NC = 4080;     
    
    // Initialize C matrix if not already done
    // memset(C, 0, M * N * sizeof(float));
    
    // L3 cache blocking loop
    for (int jc = 0; jc < N; jc += NC) {
        int nc = (jc + NC < N) ? NC : N - jc;
        
        for (int pc = 0; pc < K; pc += MC) {
            int kc = (pc + MC < K) ? MC : K - pc;
            
            // L2 cache blocking
            for (int ic = 0; ic < M; ic += TILE_M) {
                int mc = (ic + TILE_M < M) ? TILE_M : M - ic;
                
                for (int jr = jc; jr < jc + nc; jr += TILE_N) {
                    int nr = (jr + TILE_N < jc + nc) ? TILE_N : jc + nc - jr;
                    
                    for (int kr = pc; kr < pc + kc; kr += TILE_K) {
                        int kk = (kr + TILE_K < pc + kc) ? TILE_K : pc + kc - kr;
                        
                        // Micro-kernel with AVX2 optimization
                        for (int i = ic; i < ic + mc; i += 4) {
                            for (int j = jr; j < jr + nr; j += 16) {
                                
                                // Load C values (4x16 block)
                                __m256 c00 = _mm256_loadu_ps(&C[i * N + j]);
                                __m256 c01 = _mm256_loadu_ps(&C[i * N + j + 8]);
                                __m256 c10 = _mm256_loadu_ps(&C[(i+1) * N + j]);
                                __m256 c11 = _mm256_loadu_ps(&C[(i+1) * N + j + 8]);
                                __m256 c20 = _mm256_loadu_ps(&C[(i+2) * N + j]);
                                __m256 c21 = _mm256_loadu_ps(&C[(i+2) * N + j + 8]);
                                __m256 c30 = _mm256_loadu_ps(&C[(i+3) * N + j]);
                                __m256 c31 = _mm256_loadu_ps(&C[(i+3) * N + j + 8]);
                                
                                // Inner loop with prefetching
                                for (int k = kr; k < kr + kk; k++) {
                                    
                                    // Prefetch next iteration
                                    if (k + 64 < kr + kk) {
                                        _mm_prefetch((const char*)&A[i * K + k + 64], _MM_HINT_T0);
                                        _mm_prefetch((const char*)&B[k * N + j + 64], _MM_HINT_T0);
                                    }
                                    
                                    // Broadcast A values
                                    __m256 a0 = _mm256_broadcast_ss(&A[i * K + k]);
                                    __m256 a1 = _mm256_broadcast_ss(&A[(i+1) * K + k]);
                                    __m256 a2 = _mm256_broadcast_ss(&A[(i+2) * K + k]);
                                    __m256 a3 = _mm256_broadcast_ss(&A[(i+3) * K + k]);
                                    
                                    // Load B values
                                    __m256 b0 = _mm256_loadu_ps(&B[k * N + j]);
                                    __m256 b1 = _mm256_loadu_ps(&B[k * N + j + 8]);
                                    
                                    // Fused multiply-add operations
                                    c00 = _mm256_fmadd_ps(a0, b0, c00);
                                    c01 = _mm256_fmadd_ps(a0, b1, c01);
                                    c10 = _mm256_fmadd_ps(a1, b0, c10);
                                    c11 = _mm256_fmadd_ps(a1, b1, c11);
                                    c20 = _mm256_fmadd_ps(a2, b0, c20);
                                    c21 = _mm256_fmadd_ps(a2, b1, c21);
                                    c30 = _mm256_fmadd_ps(a3, b0, c30);
                                    c31 = _mm256_fmadd_ps(a3, b1, c31);
                                }
                                
                                // Store results
                                _mm256_storeu_ps(&C[i * N + j], c00);
                                _mm256_storeu_ps(&C[i * N + j + 8], c01);
                                _mm256_storeu_ps(&C[(i+1) * N + j], c10);
                                _mm256_storeu_ps(&C[(i+1) * N + j + 8], c11);
                                _mm256_storeu_ps(&C[(i+2) * N + j], c20);
                                _mm256_storeu_ps(&C[(i+2) * N + j + 8], c21);
                                _mm256_storeu_ps(&C[(i+3) * N + j], c30);
                                _mm256_storeu_ps(&C[(i+3) * N + j + 8], c31);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Optimized vector addition with better memory access patterns
void vector_add_avx2(const float* a, const float* b, float* result, int length) {
    int i = 0;
    
    // Process 32 floats at a time (4 AVX2 vectors) for better throughput
    for (; i <= length - 32; i += 32) {
        __m256 va0 = _mm256_loadu_ps(&a[i]);
        __m256 vb0 = _mm256_loadu_ps(&b[i]);
        __m256 va1 = _mm256_loadu_ps(&a[i + 8]);
        __m256 vb1 = _mm256_loadu_ps(&b[i + 8]);
        __m256 va2 = _mm256_loadu_ps(&a[i + 16]);
        __m256 vb2 = _mm256_loadu_ps(&b[i + 16]);
        __m256 va3 = _mm256_loadu_ps(&a[i + 24]);
        __m256 vb3 = _mm256_loadu_ps(&b[i + 24]);
        
        __m256 vr0 = _mm256_add_ps(va0, vb0);
        __m256 vr1 = _mm256_add_ps(va1, vb1);
        __m256 vr2 = _mm256_add_ps(va2, vb2);
        __m256 vr3 = _mm256_add_ps(va3, vb3);
        
        _mm256_storeu_ps(&result[i], vr0);
        _mm256_storeu_ps(&result[i + 8], vr1);
        _mm256_storeu_ps(&result[i + 16], vr2);
        _mm256_storeu_ps(&result[i + 24], vr3);
    }
    
    // Handle remaining 8-element chunks
    for (; i <= length - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // Handle remaining elements
    for (; i < length; i++) {
        result[i] = a[i] + b[i];
    }
}

// Optimized vector multiplication
void vector_mul_avx2(const float* a, const float* b, float* result, int length) {
    int i = 0;
    
    // Process 32 floats at a time for better pipeline utilization
    for (; i <= length - 32; i += 32) {
        __m256 va0 = _mm256_loadu_ps(&a[i]);
        __m256 vb0 = _mm256_loadu_ps(&b[i]);
        __m256 va1 = _mm256_loadu_ps(&a[i + 8]);
        __m256 vb1 = _mm256_loadu_ps(&b[i + 8]);
        __m256 va2 = _mm256_loadu_ps(&a[i + 16]);
        __m256 vb2 = _mm256_loadu_ps(&b[i + 16]);
        __m256 va3 = _mm256_loadu_ps(&a[i + 24]);
        __m256 vb3 = _mm256_loadu_ps(&b[i + 24]);
        
        __m256 vr0 = _mm256_mul_ps(va0, vb0);
        __m256 vr1 = _mm256_mul_ps(va1, vb1);
        __m256 vr2 = _mm256_mul_ps(va2, vb2);
        __m256 vr3 = _mm256_mul_ps(va3, vb3);
        
        _mm256_storeu_ps(&result[i], vr0);
        _mm256_storeu_ps(&result[i + 8], vr1);
        _mm256_storeu_ps(&result[i + 16], vr2);
        _mm256_storeu_ps(&result[i + 24], vr3);
    }
    
    // Handle remaining 8-element chunks
    for (; i <= length - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // Handle remaining elements
    for (; i < length; i++) {
        result[i] = a[i] * b[i];
    }
}

// Optimized dot product with better accumulation strategy
float dot_product_avx2(const float* a, const float* b, int length) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    int i = 0;
    
    // Process 32 floats at a time with 4-way accumulation
    for (; i <= length - 32; i += 32) {
        __m256 va0 = _mm256_loadu_ps(&a[i]);
        __m256 vb0 = _mm256_loadu_ps(&b[i]);
        __m256 va1 = _mm256_loadu_ps(&a[i + 8]);
        __m256 vb1 = _mm256_loadu_ps(&b[i + 8]);
        __m256 va2 = _mm256_loadu_ps(&a[i + 16]);
        __m256 vb2 = _mm256_loadu_ps(&b[i + 16]);
        __m256 va3 = _mm256_loadu_ps(&a[i + 24]);
        __m256 vb3 = _mm256_loadu_ps(&b[i + 24]);
        
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);
    }
    
    // Combine the 4 partial sums
    __m256 sum = _mm256_add_ps(sum0, sum1);
    sum = _mm256_add_ps(sum, sum2);
    sum = _mm256_add_ps(sum, sum3);
    
    // Handle remaining 8-element chunks
    for (; i <= length - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal add to get final sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    lo = _mm_add_ps(lo, hi);
    
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    
    float result = _mm_cvtss_f32(lo);
    
    // Handle remaining elements
    for (; i < length; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

// Matrix-vector multiplication optimized for AMD EPYC
void matvec_avx2_epyc(const float* matrix, const float* vector, 
                      float* result, int rows, int cols) {
    
    // Process 4 rows at a time for better register utilization
    int i = 0;
    for (; i <= rows - 4; i += 4) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        
        int j = 0;
        // Process 8 columns at a time
        for (; j <= cols - 8; j += 8) {
            __m256 v = _mm256_loadu_ps(&vector[j]);
            
            __m256 m0 = _mm256_loadu_ps(&matrix[i * cols + j]);
            __m256 m1 = _mm256_loadu_ps(&matrix[(i+1) * cols + j]);
            __m256 m2 = _mm256_loadu_ps(&matrix[(i+2) * cols + j]);
            __m256 m3 = _mm256_loadu_ps(&matrix[(i+3) * cols + j]);
            
            sum0 = _mm256_fmadd_ps(m0, v, sum0);
            sum1 = _mm256_fmadd_ps(m1, v, sum1);
            sum2 = _mm256_fmadd_ps(m2, v, sum2);
            sum3 = _mm256_fmadd_ps(m3, v, sum3);
        }
        
        // Horizontal add for each row
        __m128 hi0 = _mm256_extractf128_ps(sum0, 1);
        __m128 lo0 = _mm256_castps256_ps128(sum0);
        lo0 = _mm_add_ps(lo0, hi0);
        lo0 = _mm_hadd_ps(lo0, lo0);
        lo0 = _mm_hadd_ps(lo0, lo0);
        result[i] = _mm_cvtss_f32(lo0);
        
        __m128 hi1 = _mm256_extractf128_ps(sum1, 1);
        __m128 lo1 = _mm256_castps256_ps128(sum1);
        lo1 = _mm_add_ps(lo1, hi1);
        lo1 = _mm_hadd_ps(lo1, lo1);
        lo1 = _mm_hadd_ps(lo1, lo1);
        result[i+1] = _mm_cvtss_f32(lo1);
        
        __m128 hi2 = _mm256_extractf128_ps(sum2, 1);
        __m128 lo2 = _mm256_castps256_ps128(sum2);
        lo2 = _mm_add_ps(lo2, hi2);
        lo2 = _mm_hadd_ps(lo2, lo2);
        lo2 = _mm_hadd_ps(lo2, lo2);
        result[i+2] = _mm_cvtss_f32(lo2);
        
        __m128 hi3 = _mm256_extractf128_ps(sum3, 1);
        __m128 lo3 = _mm256_castps256_ps128(sum3);
        lo3 = _mm_add_ps(lo3, hi3);
        lo3 = _mm_hadd_ps(lo3, lo3);
        lo3 = _mm_hadd_ps(lo3, lo3);
        result[i+3] = _mm_cvtss_f32(lo3);
        
        // Handle remaining columns for these 4 rows
        for (; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
            result[i+1] += matrix[(i+1) * cols + j] * vector[j];
            result[i+2] += matrix[(i+2) * cols + j] * vector[j];
            result[i+3] += matrix[(i+3) * cols + j] * vector[j];
        }
    }
    
    // Handle remaining rows
    for (; i < rows; i++) {
        __m256 sum = _mm256_setzero_ps();
        int j = 0;
        
        // Process 8 elements at a time
        for (; j <= cols - 8; j += 8) {
            __m256 m = _mm256_loadu_ps(&matrix[i * cols + j]);
            __m256 v = _mm256_loadu_ps(&vector[j]);
            sum = _mm256_fmadd_ps(m, v, sum);
        }
        
        // Horizontal add
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        
        float row_sum = _mm_cvtss_f32(lo);
        
        // Handle remaining elements
        for (; j < cols; j++) {
            row_sum += matrix[i * cols + j] * vector[j];
        }
        
        result[i] = row_sum;
    }
}

// Specialized kernel for small matrices (< 512x512) common in attention mechanisms
void gemm_small_avx2(const float* A, const float* B, float* C, 
                     int M, int N, int K) {
    
    // For small matrices, use a simpler but more efficient approach
    // Optimized for typical attention head sizes (64, 96, 128)
    
    if (M <= 4 && N <= 16 && K <= 512) {
        // Micro-kernel for very small matrices
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j += 8) {
                __m256 sum = _mm256_setzero_ps();
                
                int k = 0;
                for (; k <= K - 8; k += 8) {
                    __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                    __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                    sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                }
                
                // Horizontal sum
                __m128 hi = _mm256_extractf128_ps(sum, 1);
                __m128 lo = _mm256_castps256_ps128(sum);
                lo = _mm_add_ps(lo, hi);
                lo = _mm_hadd_ps(lo, lo);
                lo = _mm_hadd_ps(lo, lo);
                
                float result = _mm_cvtss_f32(lo);
                
                // Handle remaining K elements
                for (; k < K; k++) {
                    result += A[i * K + k] * B[k * N + j];
                }
                
                C[i * N + j] = result;
            }
        }
        return;
    }
    
    // For slightly larger small matrices, use optimized tiling
    const int TILE_SIZE = 32;
    
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < K; k += TILE_SIZE) {
                
                int max_i = (i + TILE_SIZE < M) ? i + TILE_SIZE : M;
                int max_j = (j + TILE_SIZE < N) ? j + TILE_SIZE : N;
                int max_k = (k + TILE_SIZE < K) ? k + TILE_SIZE : K;
                
                // Inner kernel
                for (int ii = i; ii < max_i; ii += 2) {
                    for (int jj = j; jj < max_j; jj += 8) {
                        
                        __m256 c0 = _mm256_loadu_ps(&C[ii * N + jj]);
                        __m256 c1 = (ii + 1 < max_i) ? _mm256_loadu_ps(&C[(ii+1) * N + jj]) : _mm256_setzero_ps();
                        
                        for (int kk = k; kk < max_k; kk++) {
                            __m256 a0 = _mm256_broadcast_ss(&A[ii * K + kk]);
                            __m256 a1 = (ii + 1 < max_i) ? _mm256_broadcast_ss(&A[(ii+1) * K + kk]) : _mm256_setzero_ps();
                            __m256 b = _mm256_loadu_ps(&B[kk * N + jj]);
                            
                            c0 = _mm256_fmadd_ps(a0, b, c0);
                            if (ii + 1 < max_i) {
                                c1 = _mm256_fmadd_ps(a1, b, c1);
                            }
                        }
                        
                        _mm256_storeu_ps(&C[ii * N + jj], c0);
                        if (ii + 1 < max_i) {
                            _mm256_storeu_ps(&C[(ii+1) * N + jj], c1);
                        }
                    }
                }
            }
        }
    }
}

// Batch matrix multiplication for attention (common pattern)
void batch_gemm_attention_avx2(const float* Q, const float* K, float* scores,
                               int batch_size, int seq_len, int head_dim) {
    
    // Q: (batch, seq_len, head_dim)
    // K: (batch, seq_len, head_dim) 
    // scores: (batch, seq_len, seq_len)
    
    for (int b = 0; b < batch_size; b++) {
        const float* Q_b = Q + b * seq_len * head_dim;
        const float* K_b = K + b * seq_len * head_dim;
        float* scores_b = scores + b * seq_len * seq_len;
        
        // Compute Q @ K.T for this batch
        gemm_small_avx2(Q_b, K_b, scores_b, seq_len, seq_len, head_dim);
    }
}
'''
        
        with open('simd_kernels.c', 'w') as f:
            f.write(c_source)
    
    def _compile_library(self):
        """Compile the C library with AMD EPYC optimizations."""
        import subprocess
        
        # Compilation flags optimized for AMD EPYC 7R13
        compile_cmd = [
            'gcc',
            '-shared',
            '-fPIC',
            '-O3',
            '-march=znver3',  # AMD Zen 3 architecture (EPYC 7R13)
            '-mtune=znver3',
            '-mavx2',
            '-mfma',
            '-mbmi2',
            '-mabm',          # Advanced bit manipulation
            '-msse4.2',
            '-mpopcnt',
            '-mpclmul',
            '-maes',
            '-ffast-math',
            '-funroll-loops',
            '-fprefetch-loop-arrays',
            '-ftree-vectorize',
            '-fomit-frame-pointer',
            '-flto',
            '-fno-stack-protector',  # Remove stack protection overhead
            '-fno-plt',              # Avoid PLT for better performance
            '-mprefer-vector-width=256',  # Prefer 256-bit vectors (AVX2)
            '-falign-functions=32',       # Align functions for better cache
            '-falign-loops=32',           # Align loops for better branch prediction
            '-fno-semantic-interposition', # Allow more aggressive optimizations
            '-DNDEBUG',                   # Remove debug assertions
            'simd_kernels.c',
            '-o',
            'libsimd_kernels.so'
        ]
        
        logger.info(f"Compiling with: {' '.join(compile_cmd)}")
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Compilation failed: {result.stderr}")
            raise RuntimeError(f"Failed to compile SIMD kernels: {result.stderr}")
        
        logger.info("Successfully compiled SIMD kernels")
    
    def _load_library(self):
        """Load the compiled shared library."""
        self.lib = ctypes.CDLL('./libsimd_kernels.so')
        
        # Define function signatures
        self.lib.gemm_avx2_epyc.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.POINTER(ctypes.c_float),  # C
            ctypes.c_int,  # M
            ctypes.c_int,  # N
            ctypes.c_int   # K
        ]
        self.lib.gemm_avx2_epyc.restype = None
        
        self.lib.vector_add_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # b
            ctypes.POINTER(ctypes.c_float),  # result
            ctypes.c_int   # length
        ]
        self.lib.vector_add_avx2.restype = None
        
        self.lib.vector_mul_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # b
            ctypes.POINTER(ctypes.c_float),  # result
            ctypes.c_int   # length
        ]
        self.lib.vector_mul_avx2.restype = None
        
        self.lib.dot_product_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # b
            ctypes.c_int   # length
        ]
        self.lib.dot_product_avx2.restype = ctypes.c_float
        
        self.lib.matvec_avx2_epyc.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # matrix
            ctypes.POINTER(ctypes.c_float),  # vector
            ctypes.POINTER(ctypes.c_float),  # result
            ctypes.c_int,  # rows
            ctypes.c_int   # cols
        ]
        self.lib.matvec_avx2_epyc.restype = None

        self.lib.gemm_small_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.POINTER(ctypes.c_float),  # C
            ctypes.c_int,  # M
            ctypes.c_int,  # N
            ctypes.c_int   # K
        ]
        self.lib.gemm_small_avx2.restype = None

        self.lib.batch_gemm_attention_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Q
            ctypes.POINTER(ctypes.c_float),  # K
            ctypes.POINTER(ctypes.c_float),  # scores
            ctypes.c_int,  # batch_size
            ctypes.c_int,  # seq_len
            ctypes.c_int   # head_dim
        ]
        self.lib.batch_gemm_attention_avx2.restype = None
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Optimized matrix multiplication using AVX2."""
        if not self.avx2_supported or self.lib is None:
            return np.dot(A, B)  # Fallback to NumPy
        
        # Ensure contiguous float32 arrays
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        
        M, K = A.shape
        K2, N = B.shape
        
        if K != K2:
            raise ValueError(f"Matrix dimensions don't match: {A.shape} x {B.shape}")
        
        # Initialize result matrix
        C = np.zeros((M, N), dtype=np.float32)
        
        # Get pointers to data
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call optimized kernel
        self.lib.gemm_avx2_epyc(A_ptr, B_ptr, C_ptr, M, N, K)
        
        return C
    
    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Optimized vector addition using AVX2."""
        if not self.avx2_supported or self.lib is None:
            return a + b  # Fallback to NumPy
        
        # Ensure same length and contiguous float32
        if len(a) != len(b):
            raise ValueError("Vectors must have same length")
        
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        result = np.zeros_like(a)
        
        # Get pointers
        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call optimized kernel
        self.lib.vector_add_avx2(a_ptr, b_ptr, result_ptr, len(a))
        
        return result
    
    def vector_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Optimized element-wise vector multiplication using AVX2."""
        if not self.avx2_supported or self.lib is None:
            return a * b  # Fallback to NumPy
        
        # Ensure same length and contiguous float32
        if len(a) != len(b):
            raise ValueError("Vectors must have same length")
        
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        result = np.zeros_like(a)
        
        # Get pointers
        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call optimized kernel
        self.lib.vector_mul_avx2(a_ptr, b_ptr, result_ptr, len(a))
        
        return result
    
    def dot_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """Optimized dot product using AVX2."""
        if not self.avx2_supported or self.lib is None:
            return np.dot(a, b)  # Fallback to NumPy
        
        # Ensure same length and contiguous float32
        if len(a) != len(b):
            raise ValueError("Vectors must have same length")
        
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        
        # Get pointers
        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call optimized kernel
        return self.lib.dot_product_avx2(a_ptr, b_ptr, len(a))
    
    def matrix_vector_multiply(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Optimized matrix-vector multiplication using AVX2."""
        if not self.avx2_supported or self.lib is None:
            return np.dot(matrix, vector)  # Fallback to NumPy
        
        # Ensure compatible dimensions
        rows, cols = matrix.shape
        if len(vector) != cols:
            raise ValueError(f"Matrix columns ({cols}) must match vector length ({len(vector)})")
        
        # Ensure contiguous float32 arrays
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        result = np.zeros(rows, dtype=np.float32)
        
        # Get pointers
        matrix_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vector_ptr = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call optimized kernel
        self.lib.matvec_avx2_epyc(matrix_ptr, vector_ptr, result_ptr, rows, cols)
        
        return result

    def small_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Optimized small matrix multiplication using AVX2."""
        if not self.avx2_supported or self.lib is None:
            return np.dot(A, B)  # Fallback to NumPy
        
        # Ensure contiguous float32 arrays
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        
        M, K = A.shape
        K2, N = B.shape
        
        if K != K2:
            raise ValueError(f"Matrix dimensions don't match: {A.shape} x {B.shape}")
        
        # Initialize result matrix
        C = np.zeros((M, N), dtype=np.float32)
        
        # Get pointers to data
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call optimized kernel
        self.lib.gemm_small_avx2(A_ptr, B_ptr, C_ptr, M, N, K)
        
        return C

    def batch_matrix_multiply_attention(self, Q: np.ndarray, K: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Optimized batch matrix multiplication for attention using AVX2."""
        if not self.avx2_supported or self.lib is None:
            return np.dot(Q, K.T)  # Fallback to NumPy
        
        # Ensure contiguous float32 arrays
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32)
        scores = np.ascontiguousarray(scores, dtype=np.float32)
        
        # Get batch size, sequence length, and head dimension
        batch_size, seq_len, head_dim = Q.shape
        
        # Call optimized kernel
        self.lib.batch_gemm_attention_avx2(Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                           K.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                           scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                           batch_size, seq_len, head_dim)
        
        return scores
    
    def cleanup(self):
        """Clean up compiled files."""
        import os
        for file in ['simd_kernels.c', 'libsimd_kernels.so']:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except:
                pass


# Global instance for reuse
_simd_kernels_instance = None

def get_simd_kernels() -> SIMDKernels:
    """Get or create the global SIMD kernels instance."""
    global _simd_kernels_instance
    if _simd_kernels_instance is None:
        _simd_kernels_instance = SIMDKernels()
    return _simd_kernels_instance 