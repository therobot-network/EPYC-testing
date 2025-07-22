#!/usr/bin/env python3
"""
Test script for LLaMA quantization optimizations.
Tests INT8/INT4 quantization for CPU inference on AMD EPYC.
"""

import asyncio
import time
import sys
import os
from pathlib import Path
import torch
import numpy as np
from loguru import logger

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.models.llama33_model import Llama33Model
    from app.optimizations.quantization import (
        create_quantization_config, 
        estimate_model_size, 
        print_quantization_summary
    )
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)


class QuantizationBenchmark:
    """Benchmark suite for quantization optimizations."""
    
    def __init__(self):
        self.results = {}
        self.test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about artificial intelligence.",
            "Solve this problem: What is 15 * 23 + 47?",
            "Describe the benefits of renewable energy."
        ]
    
    def test_quantization_configs(self):
        """Test different quantization configurations."""
        logger.info("🧪 Testing quantization configurations...")
        
        configs = {
            "int8_dynamic": create_quantization_config(bits=8, strategy="dynamic"),
            "int8_static": create_quantization_config(bits=8, strategy="static"),
            "int4_weight_only": create_quantization_config(bits=4, strategy="weight_only"),
        }
        
        for name, config in configs.items():
            logger.info(f"Testing {name}:")
            logger.info(f"  Bits: {config.bits}")
            logger.info(f"  Symmetric: {config.symmetric}")
            logger.info(f"  Use zero point: {config.use_zero_point}")
            logger.info(f"  Group size: {config.group_size}")
    
    def create_dummy_model(self):
        """Create a small dummy model for testing."""
        import torch.nn as nn
        
        class DummyLLaMA(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(1000, 512)
                self.layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(512, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 512)
                    ) for _ in range(4)
                ])
                self.norm = nn.LayerNorm(512)
                self.head = nn.Linear(512, 1000)
            
            def forward(self, x):
                x = self.embed(x)
                for layer in self.layers:
                    x = x + layer(x)
                x = self.norm(x)
                return self.head(x)
        
        return DummyLLaMA()
    
    def test_quantization_accuracy(self):
        """Test quantization accuracy with dummy model."""
        logger.info("🎯 Testing quantization accuracy...")
        
        # Create dummy model
        model = self.create_dummy_model()
        model.eval()
        
        # Test input
        test_input = torch.randint(0, 1000, (1, 10))
        
        # Get original output
        with torch.no_grad():
            original_output = model(test_input)
        
        # Test different quantization levels
        for bits in [8, 4]:
            config = create_quantization_config(bits=bits, strategy="weight_only")
            
            from app.optimizations.quantization import LLaMAQuantizer
            quantizer = LLaMAQuantizer(config)
            
            # Quantize model
            quantized_model = quantizer.quantize_model(model)
            
            # Get quantized output
            with torch.no_grad():
                quantized_output = quantized_model(test_input)
            
            # Compare outputs
            mse = torch.mean((original_output - quantized_output) ** 2).item()
            max_diff = torch.max(torch.abs(original_output - quantized_output)).item()
            
            logger.info(f"INT{bits} Quantization:")
            logger.info(f"  MSE: {mse:.6f}")
            logger.info(f"  Max diff: {max_diff:.6f}")
            
            # Estimate size reduction
            original_size = estimate_model_size(model)
            print_quantization_summary(original_size, bits)
    
    def test_memory_usage(self):
        """Test memory usage with different quantization levels."""
        logger.info("💾 Testing memory usage...")
        
        model = self.create_dummy_model()
        
        # Measure original memory
        original_size = estimate_model_size(model)
        logger.info(f"Original model size: {original_size['fp32_gb']:.3f} GB")
        
        # Test quantized versions
        for bits in [8, 4]:
            config = create_quantization_config(bits=bits, strategy="weight_only")
            
            from app.optimizations.quantization import LLaMAQuantizer
            quantizer = LLaMAQuantizer(config)
            
            # Quantize model
            quantized_model = quantizer.quantize_model(model)
            
            # Estimate memory usage
            quantized_key = f'int{bits}_gb'
            reduction = original_size['fp32_gb'] / original_size[quantized_key]
            
            logger.info(f"INT{bits} quantized: {original_size[quantized_key]:.3f} GB ({reduction:.1f}x reduction)")
    
    def test_inference_speed(self):
        """Test inference speed with quantization."""
        logger.info("⚡ Testing inference speed...")
        
        model = self.create_dummy_model()
        model.eval()
        
        # Test input
        test_input = torch.randint(0, 1000, (1, 10))
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Benchmark original model
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(test_input)
        original_time = time.perf_counter() - start_time
        
        logger.info(f"Original model: {original_time*10:.2f}ms per inference")
        
        # Test quantized versions
        for bits in [8, 4]:
            config = create_quantization_config(bits=bits, strategy="weight_only")
            
            from app.optimizations.quantization import LLaMAQuantizer
            quantizer = LLaMAQuantizer(config)
            
            # Quantize model
            quantized_model = quantizer.quantize_model(model)
            quantized_model.eval()
            
            # Warm up quantized model
            with torch.no_grad():
                for _ in range(10):
                    _ = quantized_model(test_input)
            
            # Benchmark quantized model
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(100):
                    _ = quantized_model(test_input)
            quantized_time = time.perf_counter() - start_time
            
            speedup = original_time / quantized_time
            logger.info(f"INT{bits} quantized: {quantized_time*10:.2f}ms per inference ({speedup:.2f}x speedup)")
    
    async def test_llama_quantization(self, model_path: str = None):
        """Test quantization with actual LLaMA model (if available)."""
        if not model_path or not Path(model_path).exists():
            logger.warning("LLaMA model path not provided or doesn't exist, skipping LLaMA tests")
            return
        
        logger.info("🦙 Testing LLaMA model quantization...")
        
        # Test different quantization configurations
        test_configs = [
            {"bits": 8, "strategy": "dynamic"},
            {"bits": 4, "strategy": "weight_only"},
        ]
        
        for config_params in test_configs:
            logger.info(f"Testing {config_params['bits']}-bit {config_params['strategy']} quantization...")
            
            # Create quantized model
            quantization_config = create_quantization_config(**config_params)
            model = Llama33Model(
                model_path=model_path,
                device="cpu",
                quantization_config=quantization_config
            )
            
            try:
                # Load model
                await model.load()
                
                # Test inference
                test_prompt = "What is artificial intelligence?"
                
                start_time = time.time()
                response = await model.predict(
                    test_prompt,
                    max_new_tokens=50,
                    temperature=0.1
                )
                inference_time = time.time() - start_time
                
                logger.info(f"Response: {response[:100]}...")
                logger.info(f"Inference time: {inference_time:.2f}s")
                
                # Get performance stats
                stats = model.get_performance_stats()
                if stats:
                    logger.info(f"Tokens/second: {stats['avg_tokens_per_second']:.1f}")
                
            except Exception as e:
                logger.error(f"Failed to test {config_params}: {e}")
            finally:
                # Clean up
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def run_all_tests(self, model_path: str = None):
        """Run all quantization tests."""
        logger.info("🚀 Starting quantization benchmark suite")
        logger.info("=" * 60)
        
        try:
            # Test configurations
            self.test_quantization_configs()
            
            logger.info("")
            
            # Test accuracy
            self.test_quantization_accuracy()
            
            logger.info("")
            
            # Test memory usage
            self.test_memory_usage()
            
            logger.info("")
            
            # Test inference speed
            self.test_inference_speed()
            
            logger.info("")
            
            # Test with actual LLaMA model if available
            if model_path:
                asyncio.run(self.test_llama_quantization(model_path))
            
            logger.info("")
            logger.info("✅ All quantization tests completed!")
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            raise


def main():
    """Run quantization tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLaMA quantization optimizations")
    parser.add_argument("--model-path", help="Path to LLaMA model (optional)")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    args = parser.parse_args()
    
    benchmark = QuantizationBenchmark()
    
    if args.quick:
        logger.info("🏃‍♂️ Running quick quantization tests...")
        benchmark.test_quantization_configs()
        benchmark.test_quantization_accuracy()
    else:
        benchmark.run_all_tests(args.model_path)


if __name__ == "__main__":
    main() 