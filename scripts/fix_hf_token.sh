#!/bin/bash

echo "🔧 HuggingFace Token Permission Fix"
echo "=================================="

echo ""
echo "The issue: Your HuggingFace token doesn't have the right permissions to access Llama 3.3."
echo ""
echo "To fix this, you need to:"
echo ""
echo "1. 📋 Go to: https://huggingface.co/settings/tokens"
echo "2. 🔑 Find your token or create a new one"
echo "3. ✅ Make sure it has these permissions:"
echo "   - Read access to contents of all public gated repos you can access"
echo "   - OR create a 'Fine-grained' token with 'Read' access to 'meta-llama/Llama-3.3-70B-Instruct'"
echo ""
echo "4. 🦙 Also make sure you've requested access to Llama 3.3:"
echo "   - Visit: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"
echo "   - Click 'Request access' if you haven't already"
echo "   - Wait for approval (usually takes a few minutes to hours)"
echo ""
echo "5. 🔄 After fixing the token, run:"
echo "   huggingface-cli login"
echo "   # Or set environment variable: export HF_TOKEN=your_new_token"
echo ""

# Check if user wants to try logging in again
read -p "Do you want to try logging in with a new token now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running huggingface-cli login..."
    huggingface-cli login
fi 