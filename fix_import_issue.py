#!/usr/bin/env python3
"""
Fix import issue for llama31_model module.
This script diagnoses and fixes common import problems.
"""

import os
import sys
import importlib.util
from pathlib import Path

def check_file_exists():
    """Check if the llama31_model.py file exists."""
    model_file = Path("app/models/llama31_model.py")
    print(f"Checking if {model_file} exists...")
    
    if model_file.exists():
        print(f"✅ File exists: {model_file.absolute()}")
        print(f"   File size: {model_file.stat().st_size} bytes")
        return True
    else:
        print(f"❌ File not found: {model_file.absolute()}")
        return False

def check_python_path():
    """Check Python path configuration."""
    print(f"\nPython path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    current_dir = Path.cwd()
    print(f"\nCurrent working directory: {current_dir}")
    
    if str(current_dir) not in sys.path:
        print("⚠️  Current directory not in Python path, adding it...")
        sys.path.insert(0, str(current_dir))
        return True
    return False

def check_init_files():
    """Check if __init__.py files exist."""
    init_files = [
        "app/__init__.py",
        "app/models/__init__.py"
    ]
    
    print(f"\nChecking __init__.py files:")
    all_exist = True
    
    for init_file in init_files:
        path = Path(init_file)
        if path.exists():
            print(f"✅ {init_file} exists")
        else:
            print(f"❌ {init_file} missing - creating it...")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Package initialization\n")
            all_exist = False
    
    return all_exist

def test_direct_import():
    """Test importing the module directly."""
    print(f"\nTesting direct import...")
    
    try:
        # Try importing the module
        import app.models.llama31_model
        print("✅ Direct import successful")
        
        # Check if the class exists
        if hasattr(app.models.llama31_model, 'Llama31Model'):
            print("✅ Llama31Model class found")
            return True
        else:
            print("❌ Llama31Model class not found in module")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_spec_import():
    """Test importing using importlib.util.spec_from_file_location."""
    print(f"\nTesting spec-based import...")
    
    try:
        model_file = Path("app/models/llama31_model.py")
        if not model_file.exists():
            print("❌ Model file not found for spec import")
            return False
            
        spec = importlib.util.spec_from_file_location("llama31_model", model_file)
        if spec is None:
            print("❌ Could not create module spec")
            return False
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'Llama31Model'):
            print("✅ Spec-based import successful")
            return True
        else:
            print("❌ Llama31Model class not found in spec-imported module")
            return False
            
    except Exception as e:
        print(f"❌ Spec import failed: {e}")
        return False

def check_syntax():
    """Check if the Python file has syntax errors."""
    print(f"\nChecking syntax...")
    
    model_file = Path("app/models/llama31_model.py")
    if not model_file.exists():
        print("❌ File not found for syntax check")
        return False
    
    try:
        with open(model_file, 'r') as f:
            source = f.read()
        
        compile(source, str(model_file), 'exec')
        print("✅ Syntax check passed")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def fix_manager_imports():
    """Fix the imports in manager.py if needed."""
    print(f"\nChecking manager.py imports...")
    
    manager_file = Path("app/models/manager.py")
    if not manager_file.exists():
        print("❌ manager.py not found")
        return False
    
    try:
        with open(manager_file, 'r') as f:
            content = f.read()
        
        # Check if the import line exists
        if "from app.models.llama31_model import Llama31Model" in content:
            print("✅ Import statement found in manager.py")
            return True
        else:
            print("⚠️  Import statement not found, this might be the issue")
            return False
            
    except Exception as e:
        print(f"❌ Error reading manager.py: {e}")
        return False

def main():
    """Main diagnostic and fix function."""
    print("🔍 Diagnosing llama31_model import issue...")
    print("=" * 50)
    
    # Run all checks
    file_exists = check_file_exists()
    path_fixed = check_python_path()
    init_files_ok = check_init_files()
    syntax_ok = check_syntax() if file_exists else False
    manager_ok = fix_manager_imports()
    direct_import_ok = test_direct_import() if file_exists and syntax_ok else False
    spec_import_ok = test_spec_import() if file_exists and syntax_ok else False
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print(f"  File exists: {'✅' if file_exists else '❌'}")
    print(f"  Python path: {'✅' if not path_fixed else '🔧 Fixed'}")
    print(f"  __init__.py files: {'✅' if init_files_ok else '🔧 Fixed'}")
    print(f"  Syntax check: {'✅' if syntax_ok else '❌'}")
    print(f"  Manager imports: {'✅' if manager_ok else '⚠️'}")
    print(f"  Direct import: {'✅' if direct_import_ok else '❌'}")
    print(f"  Spec import: {'✅' if spec_import_ok else '❌'}")
    
    if direct_import_ok:
        print("\n🎉 Import issue appears to be resolved!")
        print("Try running your application again.")
    else:
        print("\n🚨 Import issue not resolved. Possible solutions:")
        if not file_exists:
            print("  1. Ensure the llama31_model.py file is present")
            print("  2. Check if git pull completed successfully")
            print("  3. Verify file permissions")
        if not syntax_ok:
            print("  1. Fix syntax errors in llama31_model.py")
        if not manager_ok:
            print("  1. Check manager.py import statements")
        print("  2. Try running from the project root directory")
        print("  3. Ensure virtual environment is activated")
        print("  4. Try: pip install -e . (to install in development mode)")

if __name__ == "__main__":
    main() 