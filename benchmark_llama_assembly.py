#!/usr/bin/env python3
"""
Comprehensive benchmark for LLaMA 3.3 70B with assembly optimizations.
Tests real model performance improvements on AMD EPYC 7R13.
"""

import asyncio
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import torch
from loguru import logger

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.models.llama31_model import Llama31Model
    from app.optimizations.benchmark import BenchmarkSuite
    from app.optimizations.matrix_ops import get_matrix_ops
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)


class LlamaBenchmarkSuite:
    """Comprehensive benchmark suite for LLaMA with assembly optimizations."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.matrix_ops = get_matrix_ops()
        self.results = {}
        
    async def setup_model(self):
        """Setup the LLaMA model for benchmarking."""
        if not self.model_path:
            # Try to find model in common locations
            possible_paths = [
                "models/llama-3.1-8b-instruct",
                "/home/ubuntu/models/llama-3.1-8b-instruct",
                "/opt/models/llama-3.1-8b-instruct"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    self.model_path = path
                    break
            
            if not self.model_path:
                logger.error("❌ No LLaMA model found. Please specify model path.")
                return False
        
        try:
            logger.info(f"🔄 Loading LLaMA model from {self.model_path}")
            self.model = Llama31Model(self.model_path)
            await self.model.load()
            logger.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def benchmark_assembly_kernels(self) -> Dict[str, Any]:
        """Benchmark the raw assembly kernels."""
        logger.info("🔧 Benchmarking assembly kernels...")
        
        benchmark_suite = BenchmarkSuite()
        
        # Test different matrix sizes relevant to LLaMA
        sizes = [
            (512, 512, 512),    # Small attention heads
            (1024, 1024, 1024), # Medium layers
            (2048, 2048, 2048), # Large layers
            (4096, 4096, 4096), # Very large layers (if memory allows)
        ]
        
        results = {}
        for m, n, k in sizes:
            try:
                logger.info(f"Testing {m}x{n}x{k} matrix multiplication...")
                result = benchmark_suite.benchmark_matrix_multiplication([(m, n, k)])
                results[f"{m}x{n}x{k}"] = result
                
                if result and 'results' in result and result['results']:
                    speedup = result['results'][0].get('speedup', 0)
                    logger.info(f"  Speedup: {speedup:.2f}x")
            except Exception as e:
                logger.warning(f"  Failed to benchmark {m}x{n}x{k}: {e}")
        
        return results
    
    async def benchmark_model_inference(self) -> Dict[str, Any]:
        """Benchmark actual model inference performance."""
        if not self.model:
            logger.error("Model not loaded")
            return {}
        
        logger.info("🚀 Benchmarking model inference...")
        
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about artificial intelligence.",
            "Solve this math problem: What is 15 * 23 + 47?",
            "Describe the process of photosynthesis."
        ]
        
        results = {
            'with_assembly': [],
            'without_assembly': []
        }
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"Testing prompt {i+1}: {prompt[:50]}...")
            
            # Test with assembly optimizations
            try:
                start_time = time.time()
                response = await self.model.predict(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.1  # Lower temperature for consistent results
                )
                end_time = time.time()
                
                inference_time = end_time - start_time
                tokens = len(response.split())  # Rough token count
                tokens_per_second = tokens / inference_time if inference_time > 0 else 0
                
                results['with_assembly'].append({
                    'prompt': prompt,
                    'response_length': len(response),
                    'inference_time': inference_time,
                    'tokens_per_second': tokens_per_second,
                    'response': response[:200] + "..." if len(response) > 200 else response
                })
                
                logger.info(f"  With assembly: {inference_time:.2f}s, {tokens_per_second:.1f} tokens/s")
                
            except Exception as e:
                logger.error(f"Failed inference with assembly: {e}")
        
        return results
    
    def analyze_performance_gains(self, kernel_results: Dict, inference_results: Dict):
        """Analyze and report performance gains."""
        logger.info("📊 Performance Analysis")
        logger.info("=" * 50)
        
        # Assembly kernel analysis
        if kernel_results:
            logger.info("🔧 Assembly Kernel Performance:")
            total_speedups = []
            
            for size, result in kernel_results.items():
                if result and 'results' in result and result['results']:
                    speedup = result['results'][0].get('speedup', 0)
                    if speedup > 0:
                        total_speedups.append(speedup)
                        logger.info(f"  {size}: {speedup:.2f}x speedup")
            
            if total_speedups:
                avg_speedup = sum(total_speedups) / len(total_speedups)
                logger.info(f"  Average kernel speedup: {avg_speedup:.2f}x")
        
        # Model inference analysis
        if inference_results.get('with_assembly'):
            logger.info("\n🚀 Model Inference Performance:")
            
            with_assembly = inference_results['with_assembly']
            
            total_time = sum(r['inference_time'] for r in with_assembly)
            total_tokens = sum(r['tokens_per_second'] * r['inference_time'] for r in with_assembly)
            avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            
            logger.info(f"  Average inference time: {total_time/len(with_assembly):.2f}s")
            logger.info(f"  Average tokens/second: {avg_tokens_per_second:.1f}")
            
            # Show sample responses
            logger.info("\n📝 Sample Responses:")
            for i, result in enumerate(with_assembly[:2]):  # Show first 2
                logger.info(f"  Prompt {i+1}: {result['prompt']}")
                logger.info(f"  Response: {result['response']}")
                logger.info(f"  Speed: {result['tokens_per_second']:.1f} tokens/s\n")
    
    async def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        logger.info("🎯 Starting comprehensive LLaMA assembly optimization benchmark")
        logger.info("=" * 70)
        
        # Check system info
        if self.matrix_ops.avx2_supported:
            logger.info("✅ AVX2 optimizations available")
        else:
            logger.warning("⚠️ AVX2 optimizations not available")
        
        # 1. Test assembly kernels
        kernel_results = self.benchmark_assembly_kernels()
        
        # 2. Setup model
        model_loaded = await self.setup_model()
        
        inference_results = {}
        if model_loaded:
            # 3. Test model inference
            inference_results = await self.benchmark_model_inference()
        else:
            logger.warning("Skipping model inference tests (model not available)")
        
        # 4. Analyze results
        self.analyze_performance_gains(kernel_results, inference_results)
        
        # 5. Save results
        self.results = {
            'kernel_results': kernel_results,
            'inference_results': inference_results,
            'system_info': {
                'avx2_supported': self.matrix_ops.avx2_supported,
                'epyc_optimized': getattr(self.matrix_ops, 'epyc_optimized', False)
            }
        }
        
        return self.results


async def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLaMA Assembly Optimization Benchmark")
    parser.add_argument("--model-path", help="Path to LLaMA model")
    parser.add_argument("--kernels-only", action="store_true", help="Test only assembly kernels")
    args = parser.parse_args()
    
    benchmark = LlamaBenchmarkSuite(args.model_path)
    
    if args.kernels_only:
        logger.info("🔧 Running kernels-only benchmark")
        results = benchmark.benchmark_assembly_kernels()
        benchmark.analyze_performance_gains(results, {})
    else:
        results = await benchmark.run_full_benchmark()
    
    logger.info("✅ Benchmark completed!")


if __name__ == "__main__":
    asyncio.run(main()) 