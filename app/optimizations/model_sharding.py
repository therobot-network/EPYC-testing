"""
Model sharding and parallelism for LLaMA 70B on AMD EPYC 7R13.
Distributes model layers across 96 vCPUs for efficient CPU inference.
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import threading
import queue
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
from loguru import logger
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future


@dataclass
class ShardingConfig:
    """Configuration for model sharding."""
    num_shards: int = 4  # Number of model shards
    num_workers: int = 96  # Total CPU cores
    workers_per_shard: int = 24  # Workers per shard
    pipeline_stages: int = 4  # Pipeline parallelism stages
    tensor_parallel_size: int = 1  # Tensor parallelism within layers
    use_cpu_offload: bool = True  # Offload inactive layers to CPU
    memory_efficient: bool = True  # Use memory-efficient attention
    overlap_communication: bool = True  # Overlap computation and communication


class LayerShard:
    """Individual layer shard running on specific CPU cores."""
    
    def __init__(
        self, 
        shard_id: int, 
        layers: nn.ModuleList, 
        cpu_cores: List[int],
        config: ShardingConfig
    ):
        self.shard_id = shard_id
        self.layers = layers
        self.cpu_cores = cpu_cores
        self.config = config
        
        # Thread pool for this shard
        self.executor = ThreadPoolExecutor(
            max_workers=len(cpu_cores),
            thread_name_prefix=f"shard_{shard_id}"
        )
        
        # Set CPU affinity
        self._set_cpu_affinity()
        
        # Input/output queues for pipeline
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        
        # Performance tracking
        self.processing_times = []
        self.memory_usage = []
        
        logger.info(f"Initialized shard {shard_id} with {len(layers)} layers on cores {cpu_cores}")
    
    def _set_cpu_affinity(self):
        """Set CPU affinity for this shard's threads."""
        try:
            # Set affinity for current process
            p = psutil.Process()
            p.cpu_affinity(self.cpu_cores)
            logger.debug(f"Shard {self.shard_id} bound to cores {self.cpu_cores}")
        except Exception as e:
            logger.warning(f"Failed to set CPU affinity for shard {self.shard_id}: {e}")
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through this shard's layers."""
        start_time = time.perf_counter()
        
        # Process through all layers in this shard
        for layer in self.layers:
            if attention_mask is not None:
                # For transformer layers that need attention mask
                if hasattr(layer, 'self_attn'):
                    x = layer(x, attention_mask=attention_mask)[0]
                else:
                    x = layer(x)
            else:
                x = layer(x)
        
        # Track performance
        processing_time = time.perf_counter() - start_time
        self.processing_times.append(processing_time)
        
        return x
    
    def async_forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Future:
        """Asynchronous forward pass."""
        return self.executor.submit(self.forward, x, attention_mask)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics for this shard."""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'total_forwards': len(self.processing_times),
            'throughput': len(self.processing_times) / sum(self.processing_times) if self.processing_times else 0
        }


class CPUPipelineParallel:
    """Pipeline parallelism implementation for CPU."""
    
    def __init__(self, model: nn.Module, config: ShardingConfig):
        self.model = model
        self.config = config
        self.shards = []
        self.device = torch.device("cpu")
        
        # Analyze model structure
        self.layer_info = self._analyze_model_layers()
        
        # Create shards
        self._create_shards()
        
        # Pipeline management
        self.pipeline_queue = queue.Queue()
        self.result_futures = {}
        
        logger.info(f"Initialized pipeline with {len(self.shards)} shards")
    
    def _analyze_model_layers(self) -> Dict[str, Any]:
        """Analyze model structure to determine optimal sharding."""
        layer_info = {
            'total_layers': 0,
            'layer_types': {},
            'layer_sizes': [],
            'transformer_layers': []
        }
        
        # Find transformer layers (main computation)
        for name, module in self.model.named_modules():
            if 'layer' in name.lower() or 'block' in name.lower():
                if hasattr(module, 'self_attn') or hasattr(module, 'attention'):
                    layer_info['transformer_layers'].append((name, module))
                    
                    # Estimate layer size
                    param_count = sum(p.numel() for p in module.parameters())
                    layer_info['layer_sizes'].append(param_count)
        
        layer_info['total_layers'] = len(layer_info['transformer_layers'])
        
        logger.info(f"Found {layer_info['total_layers']} transformer layers")
        return layer_info
    
    def _create_shards(self):
        """Create model shards distributed across CPU cores."""
        total_layers = self.layer_info['total_layers']
        transformer_layers = self.layer_info['transformer_layers']
        
        if total_layers == 0:
            logger.error("No transformer layers found for sharding")
            return
        
        # Calculate layers per shard
        layers_per_shard = max(1, total_layers // self.config.num_shards)
        
        # Distribute CPU cores across shards
        cores_per_shard = self.config.num_workers // self.config.num_shards
        available_cores = list(range(self.config.num_workers))
        
        for shard_id in range(self.config.num_shards):
            # Determine layers for this shard
            start_layer = shard_id * layers_per_shard
            end_layer = min((shard_id + 1) * layers_per_shard, total_layers)
            
            if start_layer >= total_layers:
                break
            
            # Get layers for this shard
            shard_layers = nn.ModuleList()
            for i in range(start_layer, end_layer):
                if i < len(transformer_layers):
                    _, layer_module = transformer_layers[i]
                    shard_layers.append(layer_module)
            
            # Assign CPU cores
            shard_cores = available_cores[shard_id * cores_per_shard:(shard_id + 1) * cores_per_shard]
            
            # Create shard
            shard = LayerShard(shard_id, shard_layers, shard_cores, self.config)
            self.shards.append(shard)
            
            logger.info(f"Shard {shard_id}: layers {start_layer}-{end_layer-1} on cores {shard_cores[:4]}...{shard_cores[-4:]}")
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through pipeline."""
        if not self.shards:
            logger.error("No shards available for forward pass")
            return x
        
        # Sequential pipeline execution
        current_input = x
        
        for shard in self.shards:
            current_input = shard.forward(current_input, attention_mask)
        
        return current_input
    
    def async_forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Asynchronous pipeline forward pass."""
        if not self.shards:
            return x
        
        # Start pipeline with first shard
        futures = []
        current_input = x
        
        # Process through pipeline stages
        for i, shard in enumerate(self.shards):
            if i == 0:
                # First shard processes input directly
                future = shard.async_forward(current_input, attention_mask)
                futures.append(future)
            else:
                # Wait for previous shard and process
                previous_output = futures[i-1].result()
                future = shard.async_forward(previous_output, attention_mask)
                futures.append(future)
        
        # Get final result
        return futures[-1].result()
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics for all shards in the pipeline."""
        stats = {
            'num_shards': len(self.shards),
            'shard_stats': [],
            'total_processing_time': 0,
            'pipeline_efficiency': 0
        }
        
        total_time = 0
        for shard in self.shards:
            shard_stats = shard.get_stats()
            stats['shard_stats'].append({
                'shard_id': shard.shard_id,
                **shard_stats
            })
            
            if 'avg_processing_time' in shard_stats:
                total_time += shard_stats['avg_processing_time']
        
        stats['total_processing_time'] = total_time
        
        # Calculate pipeline efficiency (higher is better)
        if len(self.shards) > 0 and total_time > 0:
            max_shard_time = max(
                (shard.get_stats().get('avg_processing_time', 0) for shard in self.shards),
                default=0
            )
            stats['pipeline_efficiency'] = max_shard_time / (total_time / len(self.shards)) if total_time > 0 else 0
        
        return stats


class TensorParallel:
    """Tensor parallelism for splitting large matrices across cores."""
    
    def __init__(self, config: ShardingConfig):
        self.config = config
        self.num_partitions = config.tensor_parallel_size
        
    def split_linear_layer(self, layer: nn.Linear, dim: int = 0) -> List[nn.Linear]:
        """Split a linear layer across multiple partitions."""
        weight = layer.weight.data
        bias = layer.bias.data if layer.bias is not None else None
        
        # Split weight matrix
        if dim == 0:  # Split output dimension
            split_size = weight.size(0) // self.num_partitions
            weight_splits = torch.split(weight, split_size, dim=0)
            bias_splits = torch.split(bias, split_size, dim=0) if bias is not None else [None] * len(weight_splits)
        else:  # Split input dimension
            split_size = weight.size(1) // self.num_partitions
            weight_splits = torch.split(weight, split_size, dim=1)
            bias_splits = [bias] * len(weight_splits) if bias is not None else [None] * len(weight_splits)
        
        # Create split layers
        split_layers = []
        for weight_split, bias_split in zip(weight_splits, bias_splits):
            split_layer = nn.Linear(weight_split.size(1), weight_split.size(0), bias=bias_split is not None)
            split_layer.weight.data = weight_split
            if bias_split is not None:
                split_layer.bias.data = bias_split
            split_layers.append(split_layer)
        
        return split_layers
    
    def merge_outputs(self, outputs: List[torch.Tensor], dim: int = -1) -> torch.Tensor:
        """Merge outputs from tensor parallel computation."""
        return torch.cat(outputs, dim=dim)


class ShardedLLaMAModel:
    """Main sharded LLaMA model class."""
    
    def __init__(self, model: nn.Module, config: ShardingConfig):
        self.original_model = model
        self.config = config
        
        # Initialize parallelism strategies
        self.pipeline_parallel = CPUPipelineParallel(model, config)
        self.tensor_parallel = TensorParallel(config) if config.tensor_parallel_size > 1 else None
        
        # Performance tracking
        self.inference_times = []
        self.memory_snapshots = []
        
        logger.info("Initialized sharded LLaMA model")
        self._print_sharding_info()
    
    def _print_sharding_info(self):
        """Print information about the sharding configuration."""
        logger.info("🔀 Model Sharding Configuration")
        logger.info("=" * 40)
        logger.info(f"Number of shards: {self.config.num_shards}")
        logger.info(f"Workers per shard: {self.config.workers_per_shard}")
        logger.info(f"Pipeline stages: {self.config.pipeline_stages}")
        logger.info(f"Total CPU cores: {self.config.num_workers}")
        
        if self.tensor_parallel:
            logger.info(f"Tensor parallel size: {self.config.tensor_parallel_size}")
        
        # Memory information
        memory_info = psutil.virtual_memory()
        logger.info(f"Available memory: {memory_info.available / 1024**3:.1f} GB")
        logger.info(f"Memory per shard: {memory_info.available / self.config.num_shards / 1024**3:.1f} GB")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through sharded model."""
        start_time = time.perf_counter()
        
        # Get embeddings (usually not sharded)
        if hasattr(self.original_model, 'model') and hasattr(self.original_model.model, 'embed_tokens'):
            x = self.original_model.model.embed_tokens(input_ids)
        elif hasattr(self.original_model, 'embed_tokens'):
            x = self.original_model.embed_tokens(input_ids)
        else:
            # Fallback: assume input_ids are already embeddings
            x = input_ids
        
        # Pass through pipeline
        x = self.pipeline_parallel.forward(x, attention_mask)
        
        # Final layer norm and head (usually not sharded)
        if hasattr(self.original_model, 'model') and hasattr(self.original_model.model, 'norm'):
            x = self.original_model.model.norm(x)
        
        if hasattr(self.original_model, 'lm_head'):
            logits = self.original_model.lm_head(x)
        else:
            logits = x
        
        # Track performance
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)
        
        return logits
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate text using sharded model."""
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                logits = self.forward(generated, attention_mask)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1))], dim=-1)
        
        return generated
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        pipeline_stats = self.pipeline_parallel.get_pipeline_stats()
        
        stats = {
            'model_stats': {
                'total_inferences': len(self.inference_times),
                'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
                'min_inference_time': np.min(self.inference_times) if self.inference_times else 0,
                'max_inference_time': np.max(self.inference_times) if self.inference_times else 0,
                'total_time': sum(self.inference_times),
                'throughput': len(self.inference_times) / sum(self.inference_times) if self.inference_times else 0
            },
            'pipeline_stats': pipeline_stats,
            'system_stats': {
                'cpu_count': psutil.cpu_count(),
                'memory_usage': psutil.virtual_memory().percent,
                'cpu_usage': psutil.cpu_percent(interval=1)
            }
        }
        
        return stats
    
    def print_performance_summary(self):
        """Print a comprehensive performance summary."""
        stats = self.get_performance_stats()
        
        logger.info("📊 Sharded Model Performance Summary")
        logger.info("=" * 50)
        
        # Model stats
        model_stats = stats['model_stats']
        logger.info(f"Total inferences: {model_stats['total_inferences']}")
        logger.info(f"Average inference time: {model_stats['avg_inference_time']:.3f}s")
        logger.info(f"Throughput: {model_stats['throughput']:.2f} inferences/sec")
        
        # Pipeline stats
        pipeline_stats = stats['pipeline_stats']
        logger.info(f"Pipeline efficiency: {pipeline_stats['pipeline_efficiency']:.2f}")
        logger.info(f"Total pipeline time: {pipeline_stats['total_processing_time']:.3f}s")
        
        # Shard stats
        logger.info("Shard performance:")
        for shard_stat in pipeline_stats['shard_stats']:
            shard_id = shard_stat['shard_id']
            avg_time = shard_stat.get('avg_processing_time', 0)
            throughput = shard_stat.get('throughput', 0)
            logger.info(f"  Shard {shard_id}: {avg_time:.3f}s avg, {throughput:.2f} ops/sec")
        
        # System stats
        system_stats = stats['system_stats']
        logger.info(f"CPU usage: {system_stats['cpu_usage']:.1f}%")
        logger.info(f"Memory usage: {system_stats['memory_usage']:.1f}%")


def create_sharding_config(
    num_shards: int = 4,
    strategy: str = "balanced"
) -> ShardingConfig:
    """Create sharding configuration for different strategies."""
    
    total_cores = psutil.cpu_count()
    
    if strategy == "balanced":
        # Balanced approach for general use
        return ShardingConfig(
            num_shards=num_shards,
            num_workers=total_cores,
            workers_per_shard=total_cores // num_shards,
            pipeline_stages=num_shards,
            tensor_parallel_size=1,
            use_cpu_offload=True,
            memory_efficient=True,
            overlap_communication=True
        )
    elif strategy == "memory_optimized":
        # Optimize for memory usage
        return ShardingConfig(
            num_shards=min(8, num_shards),  # More shards for memory efficiency
            num_workers=total_cores,
            workers_per_shard=total_cores // min(8, num_shards),
            pipeline_stages=min(8, num_shards),
            tensor_parallel_size=1,
            use_cpu_offload=True,
            memory_efficient=True,
            overlap_communication=False  # Prioritize memory over speed
        )
    elif strategy == "compute_optimized":
        # Optimize for compute throughput
        return ShardingConfig(
            num_shards=max(2, num_shards // 2),  # Fewer shards for compute efficiency
            num_workers=total_cores,
            workers_per_shard=total_cores // max(2, num_shards // 2),
            pipeline_stages=max(2, num_shards // 2),
            tensor_parallel_size=2 if total_cores >= 32 else 1,
            use_cpu_offload=False,
            memory_efficient=False,
            overlap_communication=True
        )
    else:
        raise ValueError(f"Unknown sharding strategy: {strategy}")


# Utility functions
def estimate_memory_per_shard(model: nn.Module, num_shards: int) -> float:
    """Estimate memory usage per shard in GB."""
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4  # Assuming FP32
    total_memory_gb = (total_params * bytes_per_param) / (1024**3)
    return total_memory_gb / num_shards


def validate_sharding_config(config: ShardingConfig) -> bool:
    """Validate sharding configuration."""
    total_cores = psutil.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Check core allocation
    if config.num_workers > total_cores:
        logger.error(f"Requested {config.num_workers} workers but only {total_cores} cores available")
        return False
    
    # Check shard configuration
    if config.num_shards * config.workers_per_shard > config.num_workers:
        logger.error("Shard configuration exceeds available workers")
        return False
    
    # Check memory requirements (rough estimate)
    estimated_memory_per_shard = 10  # GB rough estimate for 70B model shard
    required_memory = config.num_shards * estimated_memory_per_shard
    
    if required_memory > total_memory_gb * 0.8:  # Leave 20% headroom
        logger.warning(f"Estimated memory requirement ({required_memory:.1f} GB) may exceed available memory ({total_memory_gb:.1f} GB)")
    
    return True 