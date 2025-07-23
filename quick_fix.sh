#!/bin/bash

echo "🔧 Quick Fix for llama31_model Import Issue"
echo "=========================================="

# Ensure we're in the right directory
if [[ ! -f "main.py" ]]; then
    echo "❌ Not in project root directory. Please cd to EPYC-testing directory first."
    exit 1
fi

# Activate virtual environment if it exists
if [[ -d ".venv" ]]; then
    echo "🔄 Activating virtual environment..."
    source .venv/bin/activate
elif [[ -d "venv" ]]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found, using system Python"
fi

# Check if the file exists
if [[ ! -f "app/models/llama31_model.py" ]]; then
    echo "❌ llama31_model.py not found!"
    echo "🔄 Trying to pull latest changes..."
    git pull
    
    if [[ ! -f "app/models/llama31_model.py" ]]; then
        echo "❌ File still not found after git pull"
        echo "📋 Available files in app/models/:"
        ls -la app/models/
        exit 1
    fi
fi

# Ensure __init__.py files exist
echo "🔄 Ensuring __init__.py files exist..."
touch app/__init__.py
touch app/models/__init__.py

# Make sure the files have proper content
echo "# Package initialization" > app/__init__.py
echo "# Models package" > app/models/__init__.py

# Test the import
echo "🧪 Testing import..."
python3 -c "
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from app.models.llama31_model import Llama31Model
    print('✅ Import successful!')
    print('✅ Llama31Model class found')
except ImportError as e:
    print(f'❌ Import failed: {e}')
    print('🔍 Checking file structure...')
    
    import os
    if os.path.exists('app/models/llama31_model.py'):
        print('✅ File exists')
        with open('app/models/llama31_model.py', 'r') as f:
            content = f.read()
            if 'class Llama31Model' in content:
                print('✅ Class definition found in file')
            else:
                print('❌ Class definition not found in file')
    else:
        print('❌ File does not exist')
    
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    echo "🎉 Import issue fixed!"
    echo "📋 You can now run: python3 main.py"
else
    echo "❌ Import issue not resolved"
    echo "🔍 Running detailed diagnostics..."
    python3 fix_import_issue.py
fi 