"""
Advanced quantization module for LLaMA models on CPU.
Implements INT8/INT4 quantization optimized for AMD EPYC architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union
from loguru import logger
import struct
import os


class QuantizationConfig:
    """Configuration for different quantization schemes."""
    
    def __init__(
        self,
        bits: int = 8,
        group_size: int = 128,
        symmetric: bool = True,
        use_zero_point: bool = True,
        optimize_for_cpu: bool = True
    ):
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.use_zero_point = use_zero_point
        self.optimize_for_cpu = optimize_for_cpu
        
        # Quantization bounds
        if symmetric:
            self.qmin = -(2 ** (bits - 1))
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1


class CPUQuantizedLinear(nn.Module):
    """CPU-optimized quantized linear layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QuantizationConfig,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Quantized weights storage
        if config.bits == 8:
            self.register_buffer('weight_int8', torch.zeros((out_features, in_features), dtype=torch.int8))
        elif config.bits == 4:
            # Pack two 4-bit values into one int8
            packed_size = (in_features + 1) // 2
            self.register_buffer('weight_int4', torch.zeros((out_features, packed_size), dtype=torch.uint8))
        
        # Quantization parameters
        self.register_buffer('weight_scales', torch.zeros(out_features))
        if config.use_zero_point:
            self.register_buffer('weight_zero_points', torch.zeros(out_features, dtype=torch.int32))
        
        # Bias
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized computation."""
        if self.config.bits == 8:
            return self._forward_int8(x)
        elif self.config.bits == 4:
            return self._forward_int4(x)
        else:
            raise ValueError(f"Unsupported quantization bits: {self.config.bits}")
    
    def _forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        """INT8 quantized forward pass."""
        # Dequantize weights
        if self.config.use_zero_point:
            weight_fp32 = (self.weight_int8.float() - self.weight_zero_points.view(-1, 1)) * self.weight_scales.view(-1, 1)
        else:
            weight_fp32 = self.weight_int8.float() * self.weight_scales.view(-1, 1)
        
        # Standard linear operation
        output = torch.nn.functional.linear(x, weight_fp32, self.bias)
        return output
    
    def _forward_int4(self, x: torch.Tensor) -> torch.Tensor:
        """INT4 quantized forward pass."""
        # Unpack 4-bit weights
        weight_int4_unpacked = self._unpack_int4_weights()
        
        # Dequantize weights
        if self.config.use_zero_point:
            weight_fp32 = (weight_int4_unpacked.float() - self.weight_zero_points.view(-1, 1)) * self.weight_scales.view(-1, 1)
        else:
            weight_fp32 = weight_int4_unpacked.float() * self.weight_scales.view(-1, 1)
        
        # Standard linear operation
        output = torch.nn.functional.linear(x, weight_fp32, self.bias)
        return output
    
    def _unpack_int4_weights(self) -> torch.Tensor:
        """Unpack 4-bit weights from packed storage."""
        # Extract high and low 4-bit values
        high_bits = (self.weight_int4 >> 4) & 0x0F
        low_bits = self.weight_int4 & 0x0F
        
        # Interleave to reconstruct original order
        unpacked = torch.zeros((self.out_features, self.in_features), dtype=torch.int8, device=self.weight_int4.device)
        unpacked[:, ::2] = low_bits[:, :unpacked.shape[1]//2 + unpacked.shape[1]%2]
        if self.in_features > 1:
            unpacked[:, 1::2] = high_bits[:, :unpacked.shape[1]//2]
        
        return unpacked


class LLaMAQuantizer:
    """Quantizer for LLaMA models optimized for CPU inference."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.calibration_data = []
        
    def calibrate(self, model: nn.Module, dataloader, num_samples: int = 128):
        """Calibrate quantization parameters using sample data."""
        logger.info(f"Calibrating quantization with {num_samples} samples...")
        
        model.eval()
        self.calibration_data.clear()
        
        # Hook to collect activations
        hooks = []
        activation_stats = {}
        
        def collect_stats(name):
            def hook(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = []
                if isinstance(output, torch.Tensor):
                    activation_stats[name].append(output.detach().cpu())
            return hook
        
        # Register hooks on linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(collect_stats(name)))
        
        # Collect calibration data
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                
                if isinstance(batch, dict):
                    _ = model(**batch)
                else:
                    _ = model(batch)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute quantization parameters
        self.quantization_params = self._compute_quantization_params(activation_stats)
        logger.info("Calibration completed")
    
    def _compute_quantization_params(self, activation_stats: Dict) -> Dict:
        """Compute optimal quantization parameters from calibration data."""
        params = {}
        
        for name, activations in activation_stats.items():
            if not activations:
                continue
                
            # Concatenate all activation samples
            all_activations = torch.cat(activations, dim=0)
            
            # Compute statistics
            if self.config.symmetric:
                abs_max = torch.max(torch.abs(all_activations))
                scale = abs_max / (2 ** (self.config.bits - 1) - 1)
                zero_point = 0
            else:
                min_val = torch.min(all_activations)
                max_val = torch.max(all_activations)
                scale = (max_val - min_val) / (2 ** self.config.bits - 1)
                zero_point = self.config.qmin - min_val / scale
                zero_point = torch.clamp(zero_point, self.config.qmin, self.config.qmax).round()
            
            params[name] = {
                'scale': scale.item(),
                'zero_point': int(zero_point) if isinstance(zero_point, torch.Tensor) else zero_point,
                'min_val': torch.min(all_activations).item(),
                'max_val': torch.max(all_activations).item()
            }
        
        return params
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Convert model to quantized version."""
        logger.info(f"Quantizing model to {self.config.bits}-bit...")
        
        quantized_model = self._replace_linear_layers(model)
        
        # Quantize weights
        self._quantize_weights(quantized_model)
        
        logger.info("Model quantization completed")
        return quantized_model
    
    def _replace_linear_layers(self, model: nn.Module) -> nn.Module:
        """Replace linear layers with quantized versions."""
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Create quantized replacement
                quantized_layer = CPUQuantizedLinear(
                    module.in_features,
                    module.out_features,
                    self.config,
                    bias=module.bias is not None
                )
                
                # Copy bias if present
                if module.bias is not None:
                    quantized_layer.bias.copy_(module.bias)
                
                setattr(model, name, quantized_layer)
            else:
                # Recursively process child modules
                self._replace_linear_layers(module)
        
        return model
    
    def _quantize_weights(self, model: nn.Module):
        """Quantize weights of all quantized linear layers."""
        for name, module in model.named_modules():
            if isinstance(module, CPUQuantizedLinear):
                # Get original weights (if available)
                if hasattr(module, '_original_weight'):
                    weight = module._original_weight
                else:
                    # Skip if no original weights available
                    continue
                
                # Quantize weights
                if self.config.bits == 8:
                    quantized_weight, scale, zero_point = self._quantize_tensor_int8(weight)
                    module.weight_int8.copy_(quantized_weight)
                elif self.config.bits == 4:
                    quantized_weight, scale, zero_point = self._quantize_tensor_int4(weight)
                    module.weight_int4.copy_(quantized_weight)
                
                # Store quantization parameters
                module.weight_scales.copy_(scale)
                if self.config.use_zero_point:
                    module.weight_zero_points.copy_(zero_point)
    
    def _quantize_tensor_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize tensor to INT8."""
        # Per-channel quantization for weights
        if tensor.dim() >= 2:
            # Compute scale per output channel
            if self.config.symmetric:
                abs_max = torch.max(torch.abs(tensor), dim=1, keepdim=True)[0]
                scale = abs_max / (2 ** (self.config.bits - 1) - 1)
                zero_point = torch.zeros_like(scale, dtype=torch.int32)
            else:
                min_val = torch.min(tensor, dim=1, keepdim=True)[0]
                max_val = torch.max(tensor, dim=1, keepdim=True)[0]
                scale = (max_val - min_val) / (2 ** self.config.bits - 1)
                zero_point = (self.config.qmin - min_val / scale).round().clamp(self.config.qmin, self.config.qmax).to(torch.int32)
        else:
            # Per-tensor quantization
            if self.config.symmetric:
                abs_max = torch.max(torch.abs(tensor))
                scale = abs_max / (2 ** (self.config.bits - 1) - 1)
                zero_point = torch.tensor(0, dtype=torch.int32)
            else:
                min_val = torch.min(tensor)
                max_val = torch.max(tensor)
                scale = (max_val - min_val) / (2 ** self.config.bits - 1)
                zero_point = torch.round(self.config.qmin - min_val / scale).clamp(self.config.qmin, self.config.qmax).to(torch.int32)
        
        # Quantize
        if self.config.use_zero_point:
            quantized = torch.round(tensor / scale + zero_point).clamp(self.config.qmin, self.config.qmax).to(torch.int8)
        else:
            quantized = torch.round(tensor / scale).clamp(self.config.qmin, self.config.qmax).to(torch.int8)
        
        return quantized, scale.squeeze(), zero_point.squeeze()
    
    def _quantize_tensor_int4(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize tensor to INT4 with packing."""
        # First quantize to 4-bit values
        quantized_int4, scale, zero_point = self._quantize_tensor_int8(tensor)
        
        # Clamp to 4-bit range
        quantized_int4 = quantized_int4.clamp(0, 15)  # 4-bit unsigned
        
        # Pack two 4-bit values into one uint8
        packed_shape = list(quantized_int4.shape)
        packed_shape[-1] = (packed_shape[-1] + 1) // 2
        
        packed = torch.zeros(packed_shape, dtype=torch.uint8, device=quantized_int4.device)
        
        # Pack pairs of values
        for i in range(0, quantized_int4.shape[-1], 2):
            low_bits = quantized_int4[..., i].to(torch.uint8)
            if i + 1 < quantized_int4.shape[-1]:
                high_bits = quantized_int4[..., i + 1].to(torch.uint8)
            else:
                high_bits = torch.zeros_like(low_bits)
            
            packed[..., i // 2] = low_bits | (high_bits << 4)
        
        return packed, scale, zero_point
    
    def save_quantized_model(self, model: nn.Module, path: str):
        """Save quantized model to disk."""
        logger.info(f"Saving quantized model to {path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict and config
        save_dict = {
            'model_state_dict': model.state_dict(),
            'quantization_config': {
                'bits': self.config.bits,
                'group_size': self.config.group_size,
                'symmetric': self.config.symmetric,
                'use_zero_point': self.config.use_zero_point,
                'optimize_for_cpu': self.config.optimize_for_cpu
            }
        }
        
        torch.save(save_dict, path)
        logger.info("Quantized model saved successfully")
    
    def load_quantized_model(self, model: nn.Module, path: str) -> nn.Module:
        """Load quantized model from disk."""
        logger.info(f"Loading quantized model from {path}")
        
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("Quantized model loaded successfully")
        return model


def create_quantization_config(
    bits: int = 8,
    strategy: str = "dynamic"
) -> QuantizationConfig:
    """Create quantization configuration for different strategies."""
    
    if strategy == "dynamic":
        # Dynamic quantization - good for CPU inference
        return QuantizationConfig(
            bits=bits,
            group_size=128,
            symmetric=True,
            use_zero_point=False,
            optimize_for_cpu=True
        )
    elif strategy == "static":
        # Static quantization - requires calibration
        return QuantizationConfig(
            bits=bits,
            group_size=128,
            symmetric=False,
            use_zero_point=True,
            optimize_for_cpu=True
        )
    elif strategy == "weight_only":
        # Weight-only quantization - fastest
        return QuantizationConfig(
            bits=bits,
            group_size=64,
            symmetric=True,
            use_zero_point=False,
            optimize_for_cpu=True
        )
    else:
        raise ValueError(f"Unknown quantization strategy: {strategy}")


# Utility functions for model size estimation
def estimate_model_size(model: nn.Module, bits: int = 32) -> Dict[str, float]:
    """Estimate model size in different formats."""
    total_params = sum(p.numel() for p in model.parameters())
    
    sizes = {
        'fp32_gb': total_params * 4 / (1024**3),
        'fp16_gb': total_params * 2 / (1024**3),
        'int8_gb': total_params * 1 / (1024**3),
        'int4_gb': total_params * 0.5 / (1024**3),
        'total_params': total_params
    }
    
    return sizes


def print_quantization_summary(original_size: Dict, quantized_bits: int):
    """Print quantization summary."""
    quantized_key = f'int{quantized_bits}_gb'
    if quantized_key in original_size:
        reduction_ratio = original_size['fp32_gb'] / original_size[quantized_key]
        
        logger.info("🔢 Quantization Summary")
        logger.info("=" * 40)
        logger.info(f"Original (FP32): {original_size['fp32_gb']:.2f} GB")
        logger.info(f"Quantized (INT{quantized_bits}): {original_size[quantized_key]:.2f} GB")
        logger.info(f"Size reduction: {reduction_ratio:.1f}x")
        logger.info(f"Memory saved: {original_size['fp32_gb'] - original_size[quantized_key]:.2f} GB")
        logger.info(f"Total parameters: {original_size['total_params']:,}") 