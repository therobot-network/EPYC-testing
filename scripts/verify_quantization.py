#!/usr/bin/env python3
"""
Verification script for LLaMA 3.3 70B half-precision quantization optimization
on AMD EPYC c6a.24xlarge instances.
"""

import asyncio
import json
import time
import psutil
import torch
from pathlib import Path
from typing import Dict, Any
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config.settings import get_settings
from app.models.llama31_model import Llama31Model
from loguru import logger

class QuantizationVerifier:
    """Verify half-precision quantization optimizations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.results = {}
        
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements for AMD EPYC optimization."""
        logger.info("Checking system requirements...")
        
        system_info = {
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "cuda_available": torch.cuda.is_available(),
            "pytorch_version": torch.__version__,
            "cpu_brand": self._get_cpu_brand(),
        }
        
        logger.info(f"CPU cores: {system_info['cpu_count']}")
        logger.info(f"Memory: {system_info['memory_gb']:.1f}GB")
        logger.info(f"CUDA available: {system_info['cuda_available']}")
        logger.info(f"PyTorch version: {system_info['pytorch_version']}")
        logger.info(f"CPU brand: {system_info['cpu_brand']}")
        
        # Verify c6a.24xlarge specifications
        if system_info['cpu_count'] != 96:
            logger.warning(f"Expected 96 vCPUs for c6a.24xlarge, got {system_info['cpu_count']}")
        
        if system_info['memory_gb'] < 180:  # Allow some margin
            logger.warning(f"Expected ~192GB memory for c6a.24xlarge, got {system_info['memory_gb']:.1f}GB")
        
        return system_info
    
    def _get_cpu_brand(self) -> str:
        """Get CPU brand information."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        except:
            pass
        return "Unknown"
    
    def check_configuration(self) -> Dict[str, Any]:
        """Verify configuration settings."""
        logger.info("Checking configuration settings...")
        
        ec2_config = self.settings._ec2_config.get('performance', {})
        
        config_status = {
            "torch_dtype": ec2_config.get("torch_dtype"),
            "use_half_precision": ec2_config.get("use_half_precision"),
            "load_in_4bit": ec2_config.get("load_in_4bit"),
            "load_in_8bit": ec2_config.get("load_in_8bit"),
            "use_mkl": ec2_config.get("use_mkl"),
            "use_avx2": ec2_config.get("use_avx2"),
            "use_avx512": ec2_config.get("use_avx512"),
            "vectorization": ec2_config.get("vectorization"),
            "torch_threads": ec2_config.get("torch_threads"),
            "batch_size": ec2_config.get("batch_size"),
            "max_new_tokens": ec2_config.get("max_new_tokens"),
        }
        
        logger.info("Configuration status:")
        for key, value in config_status.items():
            logger.info(f"  {key}: {value}")
        
        # Validate half-precision settings
        if config_status["torch_dtype"] != "float16":
            logger.error("Expected torch_dtype to be 'float16' for half-precision optimization")
        
        if not config_status["use_half_precision"]:
            logger.error("Expected use_half_precision to be True")
        
        if config_status["load_in_4bit"] or config_status["load_in_8bit"]:
            logger.warning("Quantization is enabled, which may conflict with half-precision optimization")
        
        return config_status
    
    async def test_model_loading(self, model_path: str) -> Dict[str, Any]:
        """Test model loading with half-precision optimization."""
        logger.info(f"Testing model loading: {model_path}")
        
        start_time = time.time()
        memory_before = psutil.virtual_memory().used / (1024**3)
        
        try:
            # Initialize model
            self.model = Llama31Model(model_path)
            
            # Load model
            await self.model.load()
            
            load_time = time.time() - start_time
            memory_after = psutil.virtual_memory().used / (1024**3)
            memory_used = memory_after - memory_before
            
            # Get model info
            model_info = self.model.get_model_info()
            
            loading_results = {
                "success": True,
                "load_time_seconds": load_time,
                "memory_used_gb": memory_used,
                "model_info": model_info,
            }
            
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            logger.info(f"Memory used: {memory_used:.2f}GB")
            logger.info("Model optimizations:")
            logger.info(f"  Half-precision: {model_info.get('half_precision_enabled', False)}")
            logger.info(f"  Precision type: {model_info.get('precision_type', 'unknown')}")
            logger.info(f"  AMD EPYC optimized: {model_info.get('amd_epyc_optimized', False)}")
            logger.info(f"  MKL enabled: {model_info.get('mkl_enabled', False)}")
            logger.info(f"  Vectorization: {model_info.get('vectorization_enabled', False)}")
            
            return loading_results
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "load_time_seconds": time.time() - start_time,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3) - memory_before,
            }
    
    async def test_inference_performance(self) -> Dict[str, Any]:
        """Test inference performance with different batch sizes."""
        if not self.model or not self.model.is_loaded:
            logger.error("Model not loaded, cannot test inference")
            return {"success": False, "error": "Model not loaded"}
        
        logger.info("Testing inference performance...")
        
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot learning to paint.",
        ]
        
        performance_results = {
            "success": True,
            "tests": []
        }
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"Running inference test {i}/3...")
            
            start_time = time.time()
            cpu_before = psutil.cpu_percent(interval=None)
            memory_before = psutil.virtual_memory().used / (1024**3)
            
            try:
                # Run inference
                response = await self.model.predict(prompt, max_new_tokens=128, temperature=0.7)
                
                inference_time = time.time() - start_time
                cpu_after = psutil.cpu_percent(interval=None)
                memory_after = psutil.virtual_memory().used / (1024**3)
                
                # Calculate tokens per second (approximate)
                response_tokens = len(response.split())
                tokens_per_second = response_tokens / inference_time if inference_time > 0 else 0
                
                test_result = {
                    "test_id": i,
                    "prompt_length": len(prompt.split()),
                    "response_length": response_tokens,
                    "inference_time_seconds": inference_time,
                    "tokens_per_second": tokens_per_second,
                    "cpu_usage_percent": cpu_after - cpu_before,
                    "memory_delta_gb": memory_after - memory_before,
                    "success": True
                }
                
                logger.info(f"  Inference time: {inference_time:.2f}s")
                logger.info(f"  Tokens/second: {tokens_per_second:.1f}")
                logger.info(f"  CPU usage: {cpu_after - cpu_before:.1f}%")
                
                performance_results["tests"].append(test_result)
                
            except Exception as e:
                logger.error(f"Inference test {i} failed: {str(e)}")
                test_result = {
                    "test_id": i,
                    "error": str(e),
                    "inference_time_seconds": time.time() - start_time,
                    "success": False
                }
                performance_results["tests"].append(test_result)
        
        # Calculate averages
        successful_tests = [t for t in performance_results["tests"] if t["success"]]
        if successful_tests:
            avg_inference_time = sum(t["inference_time_seconds"] for t in successful_tests) / len(successful_tests)
            avg_tokens_per_second = sum(t["tokens_per_second"] for t in successful_tests) / len(successful_tests)
            
            performance_results["averages"] = {
                "avg_inference_time_seconds": avg_inference_time,
                "avg_tokens_per_second": avg_tokens_per_second,
                "successful_tests": len(successful_tests),
                "total_tests": len(test_prompts)
            }
            
            logger.info("Performance averages:")
            logger.info(f"  Average inference time: {avg_inference_time:.2f}s")
            logger.info(f"  Average tokens/second: {avg_tokens_per_second:.1f}")
        
        return performance_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive verification report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self.results.get("system_info", {}),
            "configuration": self.results.get("configuration", {}),
            "model_loading": self.results.get("model_loading", {}),
            "performance": self.results.get("performance", {}),
            "recommendations": []
        }
        
        # Add recommendations based on results
        config = report["configuration"]
        if config.get("torch_dtype") != "float16":
            report["recommendations"].append("Set torch_dtype to 'float16' for optimal half-precision performance")
        
        if not config.get("use_half_precision"):
            report["recommendations"].append("Enable use_half_precision in EC2 configuration")
        
        if config.get("load_in_4bit") or config.get("load_in_8bit"):
            report["recommendations"].append("Disable quantization when using half-precision optimization")
        
        system = report["system_info"]
        if system.get("cpu_count", 0) < 96:
            report["recommendations"].append("Consider using c6a.24xlarge for optimal performance with 96 vCPUs")
        
        if system.get("memory_gb", 0) < 180:
            report["recommendations"].append("Ensure sufficient memory (192GB) for LLaMA 3.3 70B model")
        
        # Performance recommendations
        performance = report.get("performance", {})
        if performance.get("success") and "averages" in performance:
            avg_tps = performance["averages"].get("avg_tokens_per_second", 0)
            if avg_tps < 10:
                report["recommendations"].append("Consider optimizing threading and SIMD settings for better performance")
            elif avg_tps > 30:
                report["recommendations"].append("Excellent performance! Consider increasing batch size for higher throughput")
        
        return report
    
    async def run_full_verification(self, model_path: str) -> Dict[str, Any]:
        """Run complete verification process."""
        logger.info("Starting LLaMA 3.3 70B half-precision quantization verification...")
        
        # System requirements
        self.results["system_info"] = self.check_system_requirements()
        
        # Configuration check
        self.results["configuration"] = self.check_configuration()
        
        # Model loading test
        self.results["model_loading"] = await self.test_model_loading(model_path)
        
        # Performance tests (only if model loaded successfully)
        if self.results["model_loading"].get("success"):
            self.results["performance"] = await self.test_inference_performance()
        else:
            self.results["performance"] = {"success": False, "error": "Model loading failed"}
        
        # Cleanup
        if self.model:
            await self.model.unload()
        
        # Generate report
        report = self.generate_report()
        
        logger.info("Verification completed!")
        return report

async def main():
    """Main verification function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify LLaMA 3.3 70B half-precision optimization")
    parser.add_argument("--model-path", required=True, help="Path to LLaMA 3.3 70B model")
    parser.add_argument("--output", default="verification_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    verifier = QuantizationVerifier()
    report = await verifier.run_full_verification(args.model_path)
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Verification report saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    system_info = report["system_info"]
    print(f"System: {system_info['cpu_count']} vCPUs, {system_info['memory_gb']:.1f}GB RAM")
    print(f"CPU: {system_info['cpu_brand']}")
    
    config = report["configuration"]
    print(f"Precision: {config['torch_dtype']}")
    print(f"Half-precision enabled: {config['use_half_precision']}")
    
    model_loading = report["model_loading"]
    if model_loading["success"]:
        print(f"Model loading: SUCCESS ({model_loading['load_time_seconds']:.1f}s, {model_loading['memory_used_gb']:.1f}GB)")
    else:
        print(f"Model loading: FAILED - {model_loading.get('error', 'Unknown error')}")
    
    performance = report["performance"]
    if performance.get("success") and "averages" in performance:
        avg_tps = performance["averages"]["avg_tokens_per_second"]
        print(f"Performance: {avg_tps:.1f} tokens/second average")
    else:
        print("Performance: Not tested (model loading failed)")
    
    if report["recommendations"]:
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main()) 