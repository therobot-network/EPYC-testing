#!/usr/bin/env python3
"""
Cleanup script to free up disk space on EC2 instance.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def get_disk_usage():
    """Get current disk usage."""
    success, stdout, stderr = run_command("df -h /")
    if success:
        lines = stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 5:
                return parts[4]  # Usage percentage
    return "Unknown"

def cleanup_huggingface_cache():
    """Clean up HuggingFace cache."""
    cache_dirs = [
        Path.home() / ".cache" / "huggingface",
        Path("/tmp/huggingface"),
        Path("/var/tmp/huggingface")
    ]
    
    total_freed = 0
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                size_before = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                shutil.rmtree(cache_dir)
                print(f"✓ Removed {cache_dir}: {size_before / (1024**3):.2f} GB freed")
                total_freed += size_before
            except Exception as e:
                print(f"⚠️ Error removing {cache_dir}: {e}")
    
    return total_freed

def cleanup_pip_cache():
    """Clean up pip cache."""
    success, stdout, stderr = run_command("pip cache purge")
    if success:
        print("✓ Pip cache cleared")
    else:
        print(f"⚠️ Error clearing pip cache: {stderr}")

def cleanup_apt_cache():
    """Clean up apt cache."""
    commands = [
        "sudo apt clean",
        "sudo apt autoclean",
        "sudo apt autoremove -y"
    ]
    
    for cmd in commands:
        success, stdout, stderr = run_command(cmd)
        if success:
            print(f"✓ {cmd}")
        else:
            print(f"⚠️ Error with {cmd}: {stderr}")

def cleanup_temp_files():
    """Clean up temporary files."""
    temp_dirs = ["/tmp", "/var/tmp"]
    
    for temp_dir in temp_dirs:
        try:
            # Only remove files, not directories
            for item in Path(temp_dir).iterdir():
                if item.is_file():
                    try:
                        item.unlink()
                    except:
                        pass
            print(f"✓ Cleaned {temp_dir}")
        except Exception as e:
            print(f"⚠️ Error cleaning {temp_dir}: {e}")

def cleanup_logs():
    """Clean up old log files."""
    success, stdout, stderr = run_command("sudo journalctl --vacuum-size=100M")
    if success:
        print("✓ Journal logs cleaned")
    else:
        print(f"⚠️ Error cleaning journal logs: {stderr}")

def main():
    print("🧹 Starting disk cleanup...")
    print(f"Current disk usage: {get_disk_usage()}")
    print()
    
    # Clean up various caches and temporary files
    print("1. Cleaning HuggingFace cache...")
    cleanup_huggingface_cache()
    
    print("\n2. Cleaning pip cache...")
    cleanup_pip_cache()
    
    print("\n3. Cleaning apt cache...")
    cleanup_apt_cache()
    
    print("\n4. Cleaning temporary files...")
    cleanup_temp_files()
    
    print("\n5. Cleaning logs...")
    cleanup_logs()
    
    print(f"\n🎉 Cleanup complete!")
    print(f"New disk usage: {get_disk_usage()}")
    
    # Show largest directories
    print("\n📊 Largest directories:")
    success, stdout, stderr = run_command("du -h --max-depth=1 / 2>/dev/null | sort -hr | head -10")
    if success:
        print(stdout)

if __name__ == "__main__":
    main() 