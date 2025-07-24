#!/usr/bin/env python3
"""
Test script to verify stop token functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.models.llama2_13b_model import Llama2_13BModel


async def test_stop_tokens():
    """Test stop token functionality."""
    print("🧪 Testing stop token functionality...")
    
    # Test the _apply_stop_tokens method directly
    model = Llama2_13BModel()
    
    # Test case 1: Text with stop token
    text1 = "Hello, this is a test. Human: What do you think? Assistant: I think it's great!"
    stop_tokens1 = ["Human:", "Assistant:"]
    result1 = model._apply_stop_tokens(text1, stop_tokens1)
    print(f"✅ Test 1 - Input: '{text1}'")
    print(f"   Stop tokens: {stop_tokens1}")
    print(f"   Result: '{result1}'")
    print()
    
    # Test case 2: Text without stop token
    text2 = "This is a simple response without any stop tokens."
    stop_tokens2 = ["Human:", "Assistant:"]
    result2 = model._apply_stop_tokens(text2, stop_tokens2)
    print(f"✅ Test 2 - Input: '{text2}'")
    print(f"   Stop tokens: {stop_tokens2}")
    print(f"   Result: '{result2}'")
    print()
    
    # Test case 3: Multiple stop tokens, earliest one wins
    text3 = "First part. Assistant: middle part. Human: end part."
    stop_tokens3 = ["Human:", "Assistant:"]
    result3 = model._apply_stop_tokens(text3, stop_tokens3)
    print(f"✅ Test 3 - Input: '{text3}'")
    print(f"   Stop tokens: {stop_tokens3}")
    print(f"   Result: '{result3}'")
    print()
    
    # Test case 4: Empty stop tokens
    text4 = "This should not be truncated."
    stop_tokens4 = None
    result4 = model._apply_stop_tokens(text4, stop_tokens4)
    print(f"✅ Test 4 - Input: '{text4}'")
    print(f"   Stop tokens: {stop_tokens4}")
    print(f"   Result: '{result4}'")
    print()
    
    # Test case 5: Custom stop tokens
    text5 = "Once upon a time, there was a kingdom. THE END. And they lived happily."
    stop_tokens5 = ["THE END"]
    result5 = model._apply_stop_tokens(text5, stop_tokens5)
    print(f"✅ Test 5 - Input: '{text5}'")
    print(f"   Stop tokens: {stop_tokens5}")
    print(f"   Result: '{result5}'")
    print()
    
    print("🎉 All stop token tests completed!")


if __name__ == "__main__":
    asyncio.run(test_stop_tokens()) 