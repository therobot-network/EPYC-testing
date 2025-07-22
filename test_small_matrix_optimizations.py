#!/usr/bin/env python3
"""
Focused test for small matrix optimizations.
Tests operations where our SIMD kernels should outperform generic libraries.
"""

import time
import numpy as np
from loguru import logger
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.optimizations.simd_kernels import get_simd_kernels
    from app.optimizations.benchmark import BenchmarkSuite
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)


def test_small_matrix_operations():
    """Test small matrix operations where we should see improvements."""
    logger.info("🔬 Testing small matrix operations...")
    
    simd_kernels = get_simd_kernels()
    
    if not simd_kernels.avx2_supported:
        logger.warning("AVX2 not supported, using fallback")
        return
    
    # Test sizes that are common in transformer attention
    test_sizes = [
        (16, 16, 16),   # Very small
        (32, 32, 32),   # Small  
        (64, 64, 64),   # Medium small
        (96, 96, 96),   # Typical attention head
        (128, 128, 128), # Large small
    ]
    
    results = []
    
    for M, K, N in test_sizes:
        logger.info(f"Testing {M}x{K}x{N} matrix multiplication...")
        
        # Generate test matrices
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Test our small matrix kernel
        num_trials = 100  # More trials for small operations
        
        # Warm up
        for _ in range(10):
            _ = simd_kernels.small_matrix_multiply(A, B)
            _ = np.dot(A, B)
        
        # Benchmark our kernel
        start_time = time.perf_counter()
        for _ in range(num_trials):
            result_opt = simd_kernels.small_matrix_multiply(A, B)
        opt_time = (time.perf_counter() - start_time) / num_trials
        
        # Benchmark NumPy
        start_time = time.perf_counter()
        for _ in range(num_trials):
            result_numpy = np.dot(A, B)
        numpy_time = (time.perf_counter() - start_time) / num_trials
        
        # Check accuracy
        max_diff = np.abs(result_opt - result_numpy).max()
        relative_error = max_diff / np.abs(result_numpy).max()
        
        # Calculate metrics
        speedup = numpy_time / opt_time
        flops = 2 * M * K * N
        opt_gflops = flops / (opt_time * 1e9)
        numpy_gflops = flops / (numpy_time * 1e9)
        
        result = {
            'size': f"{M}x{K}x{N}",
            'speedup': speedup,
            'opt_time_us': opt_time * 1e6,
            'numpy_time_us': numpy_time * 1e6,
            'opt_gflops': opt_gflops,
            'numpy_gflops': numpy_gflops,
            'max_diff': max_diff,
            'relative_error': relative_error,
            'accurate': relative_error < 1e-5
        }
        
        results.append(result)
        
        status = "✅" if speedup > 1.0 else "❌"
        logger.info(f"  {status} {speedup:.2f}x speedup ({opt_time*1e6:.1f}μs vs {numpy_time*1e6:.1f}μs)")
        logger.info(f"     GFLOPS: {opt_gflops:.1f} (opt) vs {numpy_gflops:.1f} (numpy)")
        
        if not result['accurate']:
            logger.warning(f"     ⚠️ Accuracy issue: max_diff={max_diff:.2e}")
    
    return results


def test_attention_patterns():
    """Test attention-like matrix patterns."""
    logger.info("🧠 Testing attention patterns...")
    
    simd_kernels = get_simd_kernels()
    
    if not simd_kernels.avx2_supported:
        logger.warning("AVX2 not supported, skipping attention tests")
        return []
    
    # Attention patterns: Q @ K.T where Q and K are (seq_len, head_dim)
    attention_configs = [
        (16, 64),   # Very short sequence, typical head
        (32, 64),   # Short sequence  
        (64, 64),   # Medium sequence
        (64, 96),   # Medium sequence, larger head
        (128, 64),  # Long sequence (at our limit)
    ]
    
    results = []
    
    for seq_len, head_dim in attention_configs:
        logger.info(f"Testing attention: seq_len={seq_len}, head_dim={head_dim}")
        
        # Generate Q and K matrices
        Q = np.random.randn(seq_len, head_dim).astype(np.float32)
        K = np.random.randn(seq_len, head_dim).astype(np.float32)
        
        num_trials = 50
        
        # Warm up
        for _ in range(5):
            _ = simd_kernels.small_matrix_multiply(Q, K.T)
            _ = np.dot(Q, K.T)
        
        # Benchmark our kernel (Q @ K.T)
        start_time = time.perf_counter()
        for _ in range(num_trials):
            scores_opt = simd_kernels.small_matrix_multiply(Q, K.T)
        opt_time = (time.perf_counter() - start_time) / num_trials
        
        # Benchmark NumPy
        start_time = time.perf_counter()
        for _ in range(num_trials):
            scores_numpy = np.dot(Q, K.T)
        numpy_time = (time.perf_counter() - start_time) / num_trials
        
        # Check accuracy
        max_diff = np.abs(scores_opt - scores_numpy).max()
        relative_error = max_diff / np.abs(scores_numpy).max()
        
        # Calculate metrics
        speedup = numpy_time / opt_time
        flops = 2 * seq_len * seq_len * head_dim  # Q @ K.T
        opt_gflops = flops / (opt_time * 1e9)
        numpy_gflops = flops / (numpy_time * 1e9)
        
        result = {
            'config': f"s{seq_len}_h{head_dim}",
            'seq_len': seq_len,
            'head_dim': head_dim,
            'speedup': speedup,
            'opt_time_us': opt_time * 1e6,
            'numpy_time_us': numpy_time * 1e6,
            'opt_gflops': opt_gflops,
            'numpy_gflops': numpy_gflops,
            'max_diff': max_diff,
            'relative_error': relative_error,
            'accurate': relative_error < 1e-5
        }
        
        results.append(result)
        
        status = "✅" if speedup > 1.0 else "❌"
        logger.info(f"  {status} {speedup:.2f}x speedup ({opt_time*1e6:.1f}μs vs {numpy_time*1e6:.1f}μs)")
        logger.info(f"     GFLOPS: {opt_gflops:.1f} (opt) vs {numpy_gflops:.1f} (numpy)")
        
        if not result['accurate']:
            logger.warning(f"     ⚠️ Accuracy issue: max_diff={max_diff:.2e}")
    
    return results


def main():
    """Run focused small matrix optimization tests."""
    logger.info("🎯 Starting focused small matrix optimization tests")
    logger.info("=" * 60)
    
    # Test small matrix operations
    matrix_results = test_small_matrix_operations()
    
    logger.info("")
    
    # Test attention patterns
    attention_results = test_attention_patterns()
    
    # Summary
    logger.info("")
    logger.info("📊 Summary")
    logger.info("=" * 30)
    
    if matrix_results:
        matrix_speedups = [r['speedup'] for r in matrix_results]
        matrix_wins = sum(1 for s in matrix_speedups if s > 1.0)
        logger.info(f"Matrix Operations: {matrix_wins}/{len(matrix_results)} wins")
        logger.info(f"  Average speedup: {np.mean(matrix_speedups):.2f}x")
        logger.info(f"  Best speedup: {np.max(matrix_speedups):.2f}x")
    
    if attention_results:
        attention_speedups = [r['speedup'] for r in attention_results]
        attention_wins = sum(1 for s in attention_speedups if s > 1.0)
        logger.info(f"Attention Patterns: {attention_wins}/{len(attention_results)} wins")
        logger.info(f"  Average speedup: {np.mean(attention_speedups):.2f}x")
        logger.info(f"  Best speedup: {np.max(attention_speedups):.2f}x")
    
    # Overall assessment
    all_speedups = []
    if matrix_results:
        all_speedups.extend([r['speedup'] for r in matrix_results])
    if attention_results:
        all_speedups.extend([r['speedup'] for r in attention_results])
    
    if all_speedups:
        overall_wins = sum(1 for s in all_speedups if s > 1.0)
        logger.info(f"Overall: {overall_wins}/{len(all_speedups)} operations showed improvement")
        
        if overall_wins > len(all_speedups) // 2:
            logger.info("✅ Optimizations are working for small matrices!")
        else:
            logger.warning("⚠️ Optimizations need more work")
    
    logger.info("🏁 Test completed!")


if __name__ == "__main__":
    main() 