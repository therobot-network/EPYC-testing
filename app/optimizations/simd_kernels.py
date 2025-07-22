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

// Matrix multiplication kernel using AVX2 and FMA
// Optimized for AMD EPYC 7R13 with register blocking
void gemm_avx2_epyc(const float* A, const float* B, float* C, 
                    int M, int N, int K) {
    
    // Cache-friendly tile sizes for AMD EPYC L3 cache (256MB shared)
    const int TILE_M = 64;
    const int TILE_N = 64;
    const int TILE_K = 256;
    
    // Process in tiles for cache efficiency
    for (int i = 0; i < M; i += TILE_M) {
        for (int j = 0; j < N; j += TILE_N) {
            for (int k = 0; k < K; k += TILE_K) {
                
                int max_i = (i + TILE_M < M) ? i + TILE_M : M;
                int max_j = (j + TILE_N < N) ? j + TILE_N : N;
                int max_k = (k + TILE_K < K) ? k + TILE_K : K;
                
                // Inner kernel with AVX2 optimization
                for (int ii = i; ii < max_i; ii += 4) {
                    for (int jj = j; jj < max_j; jj += 8) {
                        
                        // Load C values
                        __m256 c0 = _mm256_loadu_ps(&C[ii * N + jj]);
                        __m256 c1 = _mm256_loadu_ps(&C[(ii+1) * N + jj]);
                        __m256 c2 = _mm256_loadu_ps(&C[(ii+2) * N + jj]);
                        __m256 c3 = _mm256_loadu_ps(&C[(ii+3) * N + jj]);
                        
                        // Inner loop over K dimension
                        for (int kk = k; kk < max_k; kk++) {
                            
                            // Broadcast A values
                            __m256 a0 = _mm256_broadcast_ss(&A[ii * K + kk]);
                            __m256 a1 = _mm256_broadcast_ss(&A[(ii+1) * K + kk]);
                            __m256 a2 = _mm256_broadcast_ss(&A[(ii+2) * K + kk]);
                            __m256 a3 = _mm256_broadcast_ss(&A[(ii+3) * K + kk]);
                            
                            // Load B values
                            __m256 b = _mm256_loadu_ps(&B[kk * N + jj]);
                            
                            // Fused multiply-add operations
                            c0 = _mm256_fmadd_ps(a0, b, c0);
                            c1 = _mm256_fmadd_ps(a1, b, c1);
                            c2 = _mm256_fmadd_ps(a2, b, c2);
                            c3 = _mm256_fmadd_ps(a3, b, c3);
                        }
                        
                        // Store results
                        _mm256_storeu_ps(&C[ii * N + jj], c0);
                        _mm256_storeu_ps(&C[(ii+1) * N + jj], c1);
                        _mm256_storeu_ps(&C[(ii+2) * N + jj], c2);
                        _mm256_storeu_ps(&C[(ii+3) * N + jj], c3);
                    }
                }
            }
        }
    }
}

// Vector addition using AVX2
void vector_add_avx2(const float* a, const float* b, float* result, int length) {
    int i = 0;
    
    // Process 8 floats at a time with AVX2
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

// Vector multiplication using AVX2
void vector_mul_avx2(const float* a, const float* b, float* result, int length) {
    int i = 0;
    
    // Process 8 floats at a time with AVX2
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

// Dot product using AVX2 with horizontal add
float dot_product_avx2(const float* a, const float* b, int length) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    
    // Process 8 floats at a time
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
    
    for (int i = 0; i < rows; i++) {
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
            '-ffast-math',
            '-funroll-loops',
            '-flto',
            '-mprefer-vector-width=256',  # Prefer 256-bit vectors (AVX2)
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