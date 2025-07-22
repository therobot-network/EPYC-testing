#!/usr/bin/env python3
"""
Test script for assembly optimizations.
Run this to verify that the SIMD kernels are working correctly.
"""

import sys
import os
import numpy as np
import torch
from loguru import logger

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.optimizations.benchmark import run_quick_benchmark, BenchmarkSuite
    from app.optimizations.simd_kernels import get_simd_kernels
    from app.optimizations.matrix_ops import get_matrix_ops
except ImportError as e:
    logger.error(f"Failed to import optimization modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)


def test_simd_kernels():
    """Test basic SIMD kernel functionality."""
    logger.info("Testing SIMD kernels...")
    
    kernels = get_simd_kernels()
    
    logger.info(f"AVX2 supported: {kernels.avx2_supported}")
    logger.info(f"EPYC processor: {kernels.epyc_optimized}")
    logger.info(f"Library loaded: {kernels.lib is not None}")
    
    if not kernels.avx2_supported:
        logger.warning("AVX2 not supported - using fallback implementations")
        return True
    
    if kernels.lib is None:
        logger.error("Failed to compile/load SIMD library")
        return False
    
    # Test matrix multiplication
    logger.info("Testing matrix multiplication...")
    A = np.random.randn(256, 256).astype(np.float32)
    B = np.random.randn(256, 256).astype(np.float32)
    
    try:
        result_opt = kernels.matrix_multiply(A, B)
        result_numpy = np.dot(A, B)
        
        max_diff = np.abs(result_opt - result_numpy).max()
        relative_error = max_diff / np.abs(result_numpy).max()
        
        logger.info(f"Matrix multiply - Max diff: {max_diff:.2e}, Relative error: {relative_error:.2e}")
        
        if relative_error > 1e-5:
            logger.error("Matrix multiplication accuracy test failed!")
            return False
            
    except Exception as e:
        logger.error(f"Matrix multiplication test failed: {e}")
        return False
    
    # Test vector operations
    logger.info("Testing vector operations...")
    a = np.random.randn(10000).astype(np.float32)
    b = np.random.randn(10000).astype(np.float32)
    
    try:
        # Vector addition
        result_opt = kernels.vector_add(a, b)
        result_numpy = a + b
        max_diff = np.abs(result_opt - result_numpy).max()
        logger.info(f"Vector add - Max diff: {max_diff:.2e}")
        
        if max_diff > 1e-6:
            logger.error("Vector addition accuracy test failed!")
            return False
        
        # Dot product
        result_opt = kernels.dot_product(a, b)
        result_numpy = np.dot(a, b)
        diff = abs(result_opt - result_numpy)
        relative_error = diff / abs(result_numpy)
        logger.info(f"Dot product - Diff: {diff:.2e}, Relative error: {relative_error:.2e}")
        
        if relative_error > 1e-5:
            logger.error("Dot product accuracy test failed!")
            return False
            
    except Exception as e:
        logger.error(f"Vector operations test failed: {e}")
        return False
    
    logger.info("✅ All SIMD kernel tests passed!")
    return True


def test_matrix_ops():
    """Test PyTorch integration."""
    logger.info("Testing PyTorch matrix operations...")
    
    matrix_ops = get_matrix_ops()
    
    # Test linear layer
    logger.info("Testing linear layer...")
    x = torch.randn(2, 128, 512, dtype=torch.float32)
    weight = torch.randn(1024, 512, dtype=torch.float32)
    bias = torch.randn(1024, dtype=torch.float32)
    
    try:
        result_opt = matrix_ops.linear_forward(x, weight, bias)
        result_baseline = torch.nn.functional.linear(x, weight, bias)
        
        max_diff = torch.abs(result_opt - result_baseline).max().item()
        relative_error = max_diff / torch.abs(result_baseline).max().item()
        
        logger.info(f"Linear layer - Max diff: {max_diff:.2e}, Relative error: {relative_error:.2e}")
        
        if relative_error > 1e-4:
            logger.error("Linear layer accuracy test failed!")
            return False
            
    except Exception as e:
        logger.error(f"Linear layer test failed: {e}")
        return False
    
    # Test attention scores
    logger.info("Testing attention scores...")
    query = torch.randn(2, 64, 512, dtype=torch.float32)
    key = torch.randn(2, 64, 512, dtype=torch.float32)
    
    try:
        result_opt = matrix_ops.attention_scores(query, key)
        result_baseline = torch.matmul(query, key.transpose(-2, -1))
        
        max_diff = torch.abs(result_opt - result_baseline).max().item()
        relative_error = max_diff / torch.abs(result_baseline).max().item()
        
        logger.info(f"Attention scores - Max diff: {max_diff:.2e}, Relative error: {relative_error:.2e}")
        
        if relative_error > 1e-4:
            logger.error("Attention scores accuracy test failed!")
            return False
            
    except Exception as e:
        logger.error(f"Attention scores test failed: {e}")
        return False
    
    logger.info("✅ All PyTorch integration tests passed!")
    return True


def test_performance():
    """Test performance improvements."""
    logger.info("Testing performance improvements...")
    
    try:
        speedup, max_diff = run_quick_benchmark()
        
        logger.info(f"Quick benchmark results:")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Max difference: {max_diff:.2e}")
        
        if max_diff > 1e-5:
            logger.error("Performance test accuracy failed!")
            return False
        
        if speedup > 1.1:
            logger.info("✅ Performance improvement detected!")
        elif speedup > 0.9:
            logger.info("✅ Performance is comparable to baseline")
        else:
            logger.warning("⚠️ Performance regression detected")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return False


def run_comprehensive_benchmark():
    """Run the full benchmark suite."""
    logger.info("Running comprehensive benchmark suite...")
    
    try:
        benchmark = BenchmarkSuite()
        results = benchmark.run_full_benchmark()
        
        logger.info("Benchmark Results Summary:")
        logger.info("=" * 50)
        
        # System info
        system_info = results['system_info']
        logger.info(f"Processor: {system_info.get('processor', 'Unknown')}")
        logger.info(f"CPU cores: {system_info.get('cpu_count', 'Unknown')}")
        logger.info(f"Memory: {system_info.get('memory_gb', 0):.1f} GB")
        logger.info(f"AVX2 supported: {system_info.get('avx2_supported', False)}")
        logger.info(f"EPYC processor: {system_info.get('epyc_processor', False)}")
        
        # Overall summary
        overall = results['overall_summary']
        logger.info(f"\nOverall Results:")
        logger.info(f"Total tests: {overall.get('total_tests', 0)}")
        logger.info(f"Average speedup: {overall.get('overall_avg_speedup', 1.0):.2f}x")
        logger.info(f"Accuracy rate: {overall.get('overall_accuracy_rate', 1.0):.1%}")
        logger.info(f"AVX2 working: {overall.get('avx2_working', False)}")
        logger.info(f"Recommendation: {overall.get('recommendation', 'Unknown')}")
        
        # Category summaries
        for category, data in results['benchmark_results'].items():
            if 'summary' in data:
                summary = data['summary']
                logger.info(f"\n{category.replace('_', ' ').title()}:")
                logger.info(f"  Tests: {summary.get('count', 0)}")
                logger.info(f"  Avg speedup: {summary.get('avg_speedup', 1.0):.2f}x")
                logger.info(f"  Min/Max speedup: {summary.get('min_speedup', 1.0):.2f}x / {summary.get('max_speedup', 1.0):.2f}x")
                logger.info(f"  All accurate: {summary.get('all_accurate', True)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Comprehensive benchmark failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🚀 Starting assembly optimization tests...")
    
    all_tests_passed = True
    
    # Test 1: Basic SIMD kernels
    if not test_simd_kernels():
        all_tests_passed = False
    
    # Test 2: PyTorch integration
    if not test_matrix_ops():
        all_tests_passed = False
    
    # Test 3: Performance
    if not test_performance():
        all_tests_passed = False
    
    if all_tests_passed:
        logger.info("✅ All basic tests passed!")
        
        # Ask user if they want to run comprehensive benchmark
        try:
            response = input("\nRun comprehensive benchmark? This may take several minutes (y/N): ")
            if response.lower() in ['y', 'yes']:
                run_comprehensive_benchmark()
        except KeyboardInterrupt:
            logger.info("\nBenchmark cancelled by user")
        
        logger.info("\n🎉 Assembly optimization testing complete!")
        logger.info("Your optimizations are ready to integrate with the LLaMA model!")
        
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main() 