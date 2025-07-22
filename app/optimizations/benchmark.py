"""
Benchmark suite for testing assembly optimizations against PyTorch baselines.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from loguru import logger
import psutil
import platform

from .simd_kernels import get_simd_kernels
from .matrix_ops import get_matrix_ops


class BenchmarkSuite:
    """Comprehensive benchmark suite for assembly optimizations."""
    
    def __init__(self):
        self.simd_kernels = get_simd_kernels()
        self.matrix_ops = get_matrix_ops()
        self.results = []
        
    def benchmark_matrix_multiplication(self, sizes: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """
        Benchmark matrix multiplication across different sizes.
        
        Args:
            sizes: List of (M, K, N) tuples for matrix dimensions
        """
        logger.info("Benchmarking matrix multiplication...")
        results = []
        
        for M, K, N in sizes:
            logger.info(f"Testing matrix multiplication: ({M}, {K}) x ({K}, {N})")
            
            # Generate test matrices
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            
            # Benchmark optimized version
            start_time = time.perf_counter()
            if self.simd_kernels.avx2_supported:
                result_opt = self.simd_kernels.matrix_multiply(A, B)
            else:
                result_opt = np.dot(A, B)
            opt_time = time.perf_counter() - start_time
            
            # Benchmark NumPy baseline
            start_time = time.perf_counter()
            result_numpy = np.dot(A, B)
            numpy_time = time.perf_counter() - start_time
            
            # Check accuracy
            max_diff = np.abs(result_opt - result_numpy).max()
            relative_error = max_diff / np.abs(result_numpy).max()
            
            # Calculate performance metrics
            flops = 2 * M * K * N  # Multiply-add operations
            opt_gflops = flops / (opt_time * 1e9)
            numpy_gflops = flops / (numpy_time * 1e9)
            speedup = numpy_time / opt_time
            
            result = {
                'operation': 'matrix_multiply',
                'size': f"{M}x{K}x{N}",
                'M': M, 'K': K, 'N': N,
                'optimized_time': opt_time,
                'numpy_time': numpy_time,
                'speedup': speedup,
                'opt_gflops': opt_gflops,
                'numpy_gflops': numpy_gflops,
                'max_diff': max_diff,
                'relative_error': relative_error,
                'accuracy_ok': relative_error < 1e-5
            }
            
            results.append(result)
            logger.info(f"  Speedup: {speedup:.2f}x, GFLOPS: {opt_gflops:.2f} (opt) vs {numpy_gflops:.2f} (numpy)")
        
        return {
            'operation': 'matrix_multiplication',
            'results': results,
            'summary': self._summarize_results(results)
        }
    
    def benchmark_vector_operations(self, sizes: List[int]) -> Dict[str, Any]:
        """Benchmark vector operations (add, multiply, dot product)."""
        logger.info("Benchmarking vector operations...")
        results = []
        
        operations = [
            ('vector_add', lambda a, b: self.simd_kernels.vector_add(a, b), lambda a, b: a + b),
            ('vector_multiply', lambda a, b: self.simd_kernels.vector_multiply(a, b), lambda a, b: a * b),
            ('dot_product', lambda a, b: self.simd_kernels.dot_product(a, b), lambda a, b: np.dot(a, b))
        ]
        
        for size in sizes:
            logger.info(f"Testing vector operations with size {size}")
            
            # Generate test vectors
            a = np.random.randn(size).astype(np.float32)
            b = np.random.randn(size).astype(np.float32)
            
            for op_name, opt_func, baseline_func in operations:
                # Benchmark optimized version
                start_time = time.perf_counter()
                if self.simd_kernels.avx2_supported:
                    result_opt = opt_func(a, b)
                else:
                    result_opt = baseline_func(a, b)
                opt_time = time.perf_counter() - start_time
                
                # Benchmark baseline
                start_time = time.perf_counter()
                result_baseline = baseline_func(a, b)
                baseline_time = time.perf_counter() - start_time
                
                # Check accuracy
                if isinstance(result_opt, np.ndarray):
                    max_diff = np.abs(result_opt - result_baseline).max()
                    relative_error = max_diff / np.abs(result_baseline).max()
                else:
                    max_diff = abs(result_opt - result_baseline)
                    relative_error = max_diff / abs(result_baseline)
                
                speedup = baseline_time / opt_time
                
                result = {
                    'operation': op_name,
                    'size': size,
                    'optimized_time': opt_time,
                    'baseline_time': baseline_time,
                    'speedup': speedup,
                    'max_diff': max_diff,
                    'relative_error': relative_error,
                    'accuracy_ok': relative_error < 1e-5
                }
                
                results.append(result)
                logger.info(f"  {op_name}: {speedup:.2f}x speedup")
        
        return {
            'operation': 'vector_operations',
            'results': results,
            'summary': self._summarize_results(results)
        }
    
    def benchmark_pytorch_integration(self) -> Dict[str, Any]:
        """Benchmark PyTorch integration with typical LLaMA model sizes."""
        logger.info("Benchmarking PyTorch integration...")
        results = []
        
        # Typical LLaMA 70B dimensions
        test_cases = [
            # (batch_size, seq_len, hidden_dim, ff_dim)
            (1, 128, 8192, 28672),    # Small sequence
            (1, 512, 8192, 28672),    # Medium sequence  
            (1, 1024, 8192, 28672),   # Large sequence
            (4, 128, 8192, 28672),    # Small batch
        ]
        
        for batch_size, seq_len, hidden_dim, ff_dim in test_cases:
            logger.info(f"Testing PyTorch integration: batch={batch_size}, seq={seq_len}, hidden={hidden_dim}")
            
            # Generate test tensors
            x = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
            weight = torch.randn(ff_dim, hidden_dim, dtype=torch.float32)
            bias = torch.randn(ff_dim, dtype=torch.float32)
            
            # Test linear layer
            start_time = time.perf_counter()
            result_opt = self.matrix_ops.linear_forward(x, weight, bias)
            opt_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            result_baseline = torch.nn.functional.linear(x, weight, bias)
            baseline_time = time.perf_counter() - start_time
            
            # Check accuracy
            max_diff = torch.abs(result_opt - result_baseline).max().item()
            relative_error = max_diff / torch.abs(result_baseline).max().item()
            speedup = baseline_time / opt_time
            
            result = {
                'operation': 'linear_forward',
                'batch_size': batch_size,
                'seq_len': seq_len,
                'hidden_dim': hidden_dim,
                'ff_dim': ff_dim,
                'optimized_time': opt_time,
                'baseline_time': baseline_time,
                'speedup': speedup,
                'max_diff': max_diff,
                'relative_error': relative_error,
                'accuracy_ok': relative_error < 1e-4
            }
            
            results.append(result)
            logger.info(f"  Linear forward: {speedup:.2f}x speedup")
            
            # Test attention operations if sequence length is reasonable
            if seq_len <= 512:  # Avoid memory issues
                # Test attention scores
                query = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
                key = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
                
                start_time = time.perf_counter()
                scores_opt = self.matrix_ops.attention_scores(query, key)
                opt_time = time.perf_counter() - start_time
                
                start_time = time.perf_counter()
                scores_baseline = torch.matmul(query, key.transpose(-2, -1))
                baseline_time = time.perf_counter() - start_time
                
                max_diff = torch.abs(scores_opt - scores_baseline).max().item()
                relative_error = max_diff / torch.abs(scores_baseline).max().item()
                speedup = baseline_time / opt_time
                
                result = {
                    'operation': 'attention_scores',
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'hidden_dim': hidden_dim,
                    'optimized_time': opt_time,
                    'baseline_time': baseline_time,
                    'speedup': speedup,
                    'max_diff': max_diff,
                    'relative_error': relative_error,
                    'accuracy_ok': relative_error < 1e-4
                }
                
                results.append(result)
                logger.info(f"  Attention scores: {speedup:.2f}x speedup")
        
        return {
            'operation': 'pytorch_integration',
            'results': results,
            'summary': self._summarize_results(results)
        }
    
    def benchmark_memory_bandwidth(self) -> Dict[str, Any]:
        """Benchmark memory bandwidth utilization."""
        logger.info("Benchmarking memory bandwidth...")
        results = []
        
        # Test different sizes to stress memory hierarchy
        sizes = [
            (1024, 1024),      # L2 cache size
            (4096, 4096),      # L3 cache size  
            (8192, 8192),      # Beyond cache
            (16384, 16384),    # Large matrices
        ]
        
        for M, N in sizes:
            K = N  # Square-ish matrices
            logger.info(f"Testing memory bandwidth with {M}x{K}x{N} matrices")
            
            # Generate matrices
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            
            # Measure memory bandwidth
            data_size = (M * K + K * N + M * N) * 4  # bytes (float32)
            
            # Optimized version
            start_time = time.perf_counter()
            if self.simd_kernels.avx2_supported:
                result_opt = self.simd_kernels.matrix_multiply(A, B)
            else:
                result_opt = np.dot(A, B)
            opt_time = time.perf_counter() - start_time
            
            # Baseline
            start_time = time.perf_counter()
            result_baseline = np.dot(A, B)
            baseline_time = time.perf_counter() - start_time
            
            # Calculate bandwidth (GB/s)
            opt_bandwidth = data_size / (opt_time * 1e9)
            baseline_bandwidth = data_size / (baseline_time * 1e9)
            
            result = {
                'operation': 'memory_bandwidth',
                'matrix_size': f"{M}x{N}",
                'data_size_mb': data_size / 1e6,
                'optimized_time': opt_time,
                'baseline_time': baseline_time,
                'opt_bandwidth_gb_s': opt_bandwidth,
                'baseline_bandwidth_gb_s': baseline_bandwidth,
                'bandwidth_improvement': opt_bandwidth / baseline_bandwidth
            }
            
            results.append(result)
            logger.info(f"  Bandwidth: {opt_bandwidth:.2f} GB/s (opt) vs {baseline_bandwidth:.2f} GB/s (baseline)")
        
        return {
            'operation': 'memory_bandwidth',
            'results': results,
            'summary': self._summarize_results(results)
        }
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        logger.info("Starting full benchmark suite...")
        
        # System information
        system_info = {
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_gb': psutil.virtual_memory().total / 1e9,
            'avx2_supported': self.simd_kernels.avx2_supported,
            'epyc_processor': self.simd_kernels.epyc_optimized
        }
        
        logger.info(f"System: {system_info}")
        
        # Run benchmarks
        benchmark_results = {}
        
        # Matrix multiplication benchmark
        matrix_sizes = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
        ]
        benchmark_results['matrix_multiplication'] = self.benchmark_matrix_multiplication(matrix_sizes)
        
        # Vector operations benchmark
        vector_sizes = [1000, 10000, 100000, 1000000]
        benchmark_results['vector_operations'] = self.benchmark_vector_operations(vector_sizes)
        
        # PyTorch integration benchmark
        benchmark_results['pytorch_integration'] = self.benchmark_pytorch_integration()
        
        # Memory bandwidth benchmark
        benchmark_results['memory_bandwidth'] = self.benchmark_memory_bandwidth()
        
        # Overall summary
        overall_summary = self._create_overall_summary(benchmark_results)
        
        return {
            'system_info': system_info,
            'benchmark_results': benchmark_results,
            'overall_summary': overall_summary,
            'timestamp': time.time()
        }
    
    def _summarize_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Create summary statistics for a set of results."""
        if not results:
            return {}
        
        speedups = [r.get('speedup', 1.0) for r in results if 'speedup' in r]
        accuracy_ok = [r.get('accuracy_ok', True) for r in results if 'accuracy_ok' in r]
        
        return {
            'count': len(results),
            'avg_speedup': np.mean(speedups) if speedups else 1.0,
            'min_speedup': np.min(speedups) if speedups else 1.0,
            'max_speedup': np.max(speedups) if speedups else 1.0,
            'accuracy_pass_rate': np.mean(accuracy_ok) if accuracy_ok else 1.0,
            'all_accurate': all(accuracy_ok) if accuracy_ok else True
        }
    
    def _create_overall_summary(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create an overall summary of all benchmark results."""
        all_speedups = []
        all_accuracy = []
        
        for category, results in benchmark_results.items():
            if 'results' in results:
                for result in results['results']:
                    if 'speedup' in result:
                        all_speedups.append(result['speedup'])
                    if 'accuracy_ok' in result:
                        all_accuracy.append(result['accuracy_ok'])
        
        return {
            'total_tests': len(all_speedups),
            'overall_avg_speedup': np.mean(all_speedups) if all_speedups else 1.0,
            'overall_accuracy_rate': np.mean(all_accuracy) if all_accuracy else 1.0,
            'avx2_working': self.simd_kernels.avx2_supported and self.simd_kernels.lib is not None,
            'recommendation': self._get_recommendation(all_speedups, all_accuracy)
        }
    
    def _get_recommendation(self, speedups: List[float], accuracy: List[bool]) -> str:
        """Generate a recommendation based on benchmark results."""
        if not speedups:
            return "No performance tests completed"
        
        avg_speedup = np.mean(speedups)
        accuracy_rate = np.mean(accuracy) if accuracy else 1.0
        
        if accuracy_rate < 0.95:
            return "CAUTION: Accuracy issues detected. Review implementation."
        elif avg_speedup > 2.0:
            return "EXCELLENT: Significant performance improvement. Deploy optimizations."
        elif avg_speedup > 1.2:
            return "GOOD: Moderate performance improvement. Consider deploying."
        elif avg_speedup > 0.8:
            return "MARGINAL: Small performance gain. Evaluate trade-offs."
        else:
            return "POOR: Performance regression. Do not deploy."


def run_quick_benchmark():
    """Run a quick benchmark for testing."""
    benchmark = BenchmarkSuite()
    
    # Quick matrix multiplication test
    A = np.random.randn(512, 512).astype(np.float32)
    B = np.random.randn(512, 512).astype(np.float32)
    
    logger.info("Quick benchmark: 512x512 matrix multiplication")
    
    # Optimized
    start = time.perf_counter()
    if benchmark.simd_kernels.avx2_supported:
        result_opt = benchmark.simd_kernels.matrix_multiply(A, B)
    else:
        result_opt = np.dot(A, B)
    opt_time = time.perf_counter() - start
    
    # Baseline
    start = time.perf_counter()
    result_baseline = np.dot(A, B)
    baseline_time = time.perf_counter() - start
    
    speedup = baseline_time / opt_time
    max_diff = np.abs(result_opt - result_baseline).max()
    
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Max difference: {max_diff:.2e}")
    logger.info(f"AVX2 supported: {benchmark.simd_kernels.avx2_supported}")
    
    return speedup, max_diff 