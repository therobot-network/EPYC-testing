"""
Optimized matrix operations for LLaMA model integration.
Provides PyTorch-compatible wrappers around SIMD kernels.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from loguru import logger

from .simd_kernels import get_simd_kernels


class OptimizedMatrixOps:
    """Optimized matrix operations using SIMD kernels for PyTorch tensors."""
    
    def __init__(self):
        self.simd_kernels = get_simd_kernels()
        self.use_optimizations = self.simd_kernels.avx2_supported
        
        # Thresholds for when to use our optimizations vs falling back to optimized libraries
        self.small_matrix_threshold = 256  # Use SIMD for matrices smaller than this
        self.vector_threshold = 10000      # Use SIMD for vectors smaller than this
        
        if self.use_optimizations:
            logger.info("Initialized optimized matrix operations with AVX2 support")
            logger.info(f"Using SIMD for matrices < {self.small_matrix_threshold}x{self.small_matrix_threshold}")
        else:
            logger.warning("Using fallback matrix operations (no AVX2 support)")
    
    def linear_forward(self, input_tensor: torch.Tensor, weight: torch.Tensor, 
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optimized linear layer forward pass.
        Equivalent to torch.nn.functional.linear but using AVX2 optimizations.
        
        Args:
            input_tensor: Input tensor of shape (..., in_features)
            weight: Weight tensor of shape (out_features, in_features)
            bias: Optional bias tensor of shape (out_features,)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Check if we should use optimizations based on matrix size
        out_features, in_features = weight.shape
        use_simd = (self.use_optimizations and 
                   out_features <= self.small_matrix_threshold and 
                   in_features <= self.small_matrix_threshold)
        
        if not use_simd:
            return torch.nn.functional.linear(input_tensor, weight, bias)
        
        # Get original shape and flatten input for matrix multiplication
        original_shape = input_tensor.shape
        input_2d = input_tensor.view(-1, input_tensor.shape[-1])
        
        # For very small matrices, our SIMD kernels might be competitive
        batch_size = input_2d.shape[0]
        if batch_size * out_features * in_features < 1000000:  # Small operation
            # Convert to numpy for SIMD operations
            input_np = input_2d.detach().cpu().numpy().astype(np.float32)
            weight_np = weight.detach().cpu().numpy().astype(np.float32)
            
            # Perform optimized matrix multiplication: input @ weight.T
            # Note: weight is (out_features, in_features), so we need weight.T for correct multiplication
            weight_t = weight_np.T
            
            # Use specialized small matrix kernel for very small matrices
            if (out_features <= 128 and in_features <= 512 and batch_size <= 32):
                result_np = self.simd_kernels.small_matrix_multiply(input_np, weight_t)
            else:
                result_np = self.simd_kernels.matrix_multiply(input_np, weight_t)
            
            # Convert back to tensor
            result_tensor = torch.from_numpy(result_np)
            
            # Add bias if provided
            if bias is not None:
                result_tensor = result_tensor + bias.cpu()
            
            # Restore original shape (except last dimension)
            output_shape = original_shape[:-1] + (weight.shape[0],)
            result_tensor = result_tensor.view(output_shape)
            
            # Move to same device as input
            return result_tensor.to(input_tensor.device)
        else:
            # Fall back to PyTorch for larger operations
            return torch.nn.functional.linear(input_tensor, weight, bias)
    
    def attention_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Optimized attention score computation: Q @ K.T
        
        Args:
            query: Query tensor of shape (batch, seq_len, dim)
            key: Key tensor of shape (batch, seq_len, dim)
            
        Returns:
            Attention scores of shape (batch, seq_len, seq_len)
        """
        if not self.use_optimizations:
            return torch.matmul(query, key.transpose(-2, -1))
        
        batch_size, seq_len, dim = query.shape
        
        # Process each batch separately for now
        results = []
        for b in range(batch_size):
            query_b = query[b].detach().cpu().numpy().astype(np.float32)  # (seq_len, dim)
            key_b = key[b].detach().cpu().numpy().astype(np.float32)      # (seq_len, dim)
            
            # Compute Q @ K.T using optimized matrix multiplication
            scores_b = self.simd_kernels.matrix_multiply(query_b, key_b.T)
            results.append(torch.from_numpy(scores_b))
        
        # Stack results
        result_tensor = torch.stack(results)
        return result_tensor.to(query.device)
    
    def attention_output(self, attention_weights: torch.Tensor, 
                        value: torch.Tensor) -> torch.Tensor:
        """
        Optimized attention output computation: attention_weights @ V
        
        Args:
            attention_weights: Attention weights of shape (batch, seq_len, seq_len)
            value: Value tensor of shape (batch, seq_len, dim)
            
        Returns:
            Attention output of shape (batch, seq_len, dim)
        """
        if not self.use_optimizations:
            return torch.matmul(attention_weights, value)
        
        batch_size, seq_len, dim = value.shape
        
        # Process each batch separately
        results = []
        for b in range(batch_size):
            weights_b = attention_weights[b].detach().cpu().numpy().astype(np.float32)
            value_b = value[b].detach().cpu().numpy().astype(np.float32)
            
            # Compute attention_weights @ V
            output_b = self.simd_kernels.matrix_multiply(weights_b, value_b)
            results.append(torch.from_numpy(output_b))
        
        result_tensor = torch.stack(results)
        return result_tensor.to(value.device)
    
    def feedforward_forward(self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                           w3: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optimized feedforward network forward pass.
        Handles both standard FFN and SwiGLU variants.
        
        Args:
            x: Input tensor
            w1: First linear layer weight
            w2: Second linear layer weight  
            w3: Optional third linear layer weight (for SwiGLU)
            
        Returns:
            Output tensor
        """
        if w3 is not None:
            # SwiGLU: w2(silu(w1(x)) * w3(x))
            gate = self.linear_forward(x, w1)
            gate = torch.nn.functional.silu(gate)
            
            up = self.linear_forward(x, w3)
            intermediate = gate * up
            
            return self.linear_forward(intermediate, w2)
        else:
            # Standard FFN: w2(relu(w1(x)))
            intermediate = self.linear_forward(x, w1)
            intermediate = torch.nn.functional.relu(intermediate)
            return self.linear_forward(intermediate, w2)
    
    def layer_norm_forward(self, x: torch.Tensor, weight: torch.Tensor, 
                          bias: Optional[torch.Tensor] = None, 
                          eps: float = 1e-5) -> torch.Tensor:
        """
        Optimized layer normalization.
        Falls back to PyTorch for now, but could be optimized with SIMD.
        """
        # For now, use PyTorch's optimized implementation
        # Could be further optimized with SIMD operations
        return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)
    
    def embedding_forward(self, input_ids: torch.Tensor, 
                         embedding_weight: torch.Tensor) -> torch.Tensor:
        """
        Optimized embedding lookup.
        Uses PyTorch's efficient implementation.
        """
        return torch.nn.functional.embedding(input_ids, embedding_weight)
    
    def rms_norm_forward(self, x: torch.Tensor, weight: torch.Tensor, 
                        eps: float = 1e-6) -> torch.Tensor:
        """
        Optimized RMSNorm implementation.
        Used in LLaMA models instead of LayerNorm.
        """
        # Could be optimized with SIMD, but using PyTorch for now
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x
    
    def rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, 
                      sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding.
        Could be optimized with SIMD operations.
        """
        # For now, use standard implementation
        # This could benefit from SIMD optimizations for the rotation operations
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + rotated * sin
    
    def benchmark_operation(self, operation_name: str, *args, **kwargs):
        """
        Benchmark a specific operation against PyTorch baseline.
        """
        import time
        
        # Run optimized version
        start_time = time.perf_counter()
        if hasattr(self, operation_name):
            optimized_result = getattr(self, operation_name)(*args, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation_name}")
        optimized_time = time.perf_counter() - start_time
        
        # Run PyTorch baseline (if available)
        baseline_time = None
        baseline_result = None
        
        if operation_name == "linear_forward":
            start_time = time.perf_counter()
            baseline_result = torch.nn.functional.linear(*args, **kwargs)
            baseline_time = time.perf_counter() - start_time
        
        elif operation_name == "attention_scores":
            query, key = args
            start_time = time.perf_counter()
            baseline_result = torch.matmul(query, key.transpose(-2, -1))
            baseline_time = time.perf_counter() - start_time
        
        # Calculate speedup
        speedup = baseline_time / optimized_time if baseline_time else None
        
        # Check accuracy
        accuracy = None
        if baseline_result is not None:
            diff = torch.abs(optimized_result - baseline_result).max().item()
            accuracy = f"Max diff: {diff:.2e}"
        
        return {
            "operation": operation_name,
            "optimized_time": optimized_time,
            "baseline_time": baseline_time,
            "speedup": speedup,
            "accuracy": accuracy,
            "result_shape": optimized_result.shape
        }


# Global instance for reuse
_matrix_ops_instance = None

def get_matrix_ops() -> OptimizedMatrixOps:
    """Get or create the global optimized matrix operations instance."""
    global _matrix_ops_instance
    if _matrix_ops_instance is None:
        _matrix_ops_instance = OptimizedMatrixOps()
    return _matrix_ops_instance


def patch_pytorch_linear_layers():
    """
    Patch PyTorch's linear layers to use optimized matrix operations.
    This monkey-patches torch.nn.functional.linear to use our SIMD optimizations
    when appropriate (for smaller matrices where our kernels are competitive).
    """
    matrix_ops = get_matrix_ops()
    
    if not matrix_ops.use_optimizations:
        logger.warning("SIMD optimizations not available, skipping linear layer patching")
        return
    
    # Store original implementation
    if not hasattr(torch.nn.functional, '_original_linear'):
        torch.nn.functional._original_linear = torch.nn.functional.linear
        
        def optimized_linear(input_tensor: torch.Tensor, weight: torch.Tensor, 
                           bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Optimized linear layer implementation that uses SIMD kernels for small matrices.
            Falls back to original PyTorch implementation for larger matrices.
            """
            try:
                # Check if we should use our optimizations
                if len(weight.shape) == 2:  # Standard linear layer
                    out_features, in_features = weight.shape
                    
                    # Use our optimizations for small to medium matrices
                    if (out_features <= matrix_ops.small_matrix_threshold and 
                        in_features <= matrix_ops.small_matrix_threshold):
                        return matrix_ops.linear_forward(input_tensor, weight, bias)
                
                # Fall back to original implementation for large matrices or edge cases
                return torch.nn.functional._original_linear(input_tensor, weight, bias)
                
            except Exception as e:
                # If anything goes wrong, fall back to original implementation
                logger.warning(f"SIMD optimization failed, falling back to PyTorch: {e}")
                return torch.nn.functional._original_linear(input_tensor, weight, bias)
        
        # Replace the function
        torch.nn.functional.linear = optimized_linear
        logger.info("✅ PyTorch linear layers patched with SIMD optimizations")
        
    else:
        logger.info("PyTorch linear layers already patched")


def unpatch_pytorch_linear_layers():
    """
    Restore original PyTorch linear layer implementation.
    """
    if hasattr(torch.nn.functional, '_original_linear'):
        torch.nn.functional.linear = torch.nn.functional._original_linear
        delattr(torch.nn.functional, '_original_linear')
        logger.info("PyTorch linear layers restored to original implementation")
    else:
        logger.warning("No patched linear layers found to restore") 