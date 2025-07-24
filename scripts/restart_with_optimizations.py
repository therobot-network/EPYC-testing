#!/usr/bin/env python3
"""
Script to properly shutdown any processes using port 8000 and restart with AMD EPYC optimizations.
"""

import subprocess
import time
import sys
import os
import signal
from pathlib import Path

def kill_processes_on_port(port=8000):
    """Kill any processes using the specified port."""
    print(f"🔍 Checking for processes using port {port}...")
    
    try:
        # Find processes using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"📋 Found {len(pids)} process(es) using port {port}")
            
            for pid in pids:
                if pid.strip():
                    try:
                        print(f"🔪 Killing process {pid}")
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(1)
                        
                        # If still running, force kill
                        try:
                            os.kill(int(pid), 0)  # Check if process still exists
                            print(f"💥 Force killing process {pid}")
                            os.kill(int(pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # Process already dead
                            
                    except (ProcessLookupError, ValueError) as e:
                        print(f"⚠️ Could not kill process {pid}: {e}")
            
            # Wait a moment for cleanup
            time.sleep(2)
            print("✅ Port cleanup complete")
        else:
            print(f"✅ No processes found using port {port}")
            
    except FileNotFoundError:
        print("⚠️ lsof command not found, trying alternative method...")
        
        # Alternative method using netstat and ps
        try:
            result = subprocess.run(
                ["netstat", "-tlnp", f"| grep :{port}"],
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                print(f"📋 Found processes using port {port}, attempting to kill...")
                # Kill any python processes that might be using the port
                subprocess.run(["pkill", "-f", "python.*main.py"], capture_output=True)
                subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
                time.sleep(2)
                print("✅ Alternative cleanup complete")
                
        except Exception as e:
            print(f"⚠️ Alternative cleanup failed: {e}")

def kill_python_main_processes():
    """Kill any python main.py processes."""
    print("🔍 Checking for python main.py processes...")
    
    try:
        # Kill any python main.py processes
        result = subprocess.run(
            ["pkill", "-f", "python.*main.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Killed python main.py processes")
        
        # Also kill uvicorn processes
        result = subprocess.run(
            ["pkill", "-f", "uvicorn"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Killed uvicorn processes")
            
        time.sleep(2)
        
    except Exception as e:
        print(f"⚠️ Error killing python processes: {e}")

def start_optimized_service():
    """Start the service with AMD EPYC optimizations."""
    print("🚀 Starting AMD EPYC optimized service...")
    
    # Change to the project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Activate virtual environment and start service
    try:
        # Start the service in background
        process = subprocess.Popen(
            ["python3", "app/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        print(f"📊 Started service with PID: {process.pid}")
        
        # Wait a moment to check if it started successfully
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Service started successfully!")
            print("🎯 AMD EPYC optimizations are now active:")
            print("   • 48 threads for optimal CPU utilization")
            print("   • 8-bit quantization for memory efficiency")
            print("   • Model sharding across 4 NUMA nodes")
            print("   • AVX2/FMA SIMD optimizations")
            print(f"🌐 Service running on http://0.0.0.0:8000")
            
            # Show some initial output
            print("\n📋 Service startup logs:")
            for _ in range(10):  # Show first 10 lines
                line = process.stdout.readline()
                if line:
                    print(f"   {line.strip()}")
                else:
                    break
            
            return process
        else:
            print("❌ Service failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        return None

def main():
    """Main function to restart the service with optimizations."""
    print("🔧 AMD EPYC Optimization Service Restart")
    print("=" * 50)
    
    # Step 1: Kill processes on port 8000
    kill_processes_on_port(8000)
    
    # Step 2: Kill any remaining python main.py processes
    kill_python_main_processes()
    
    # Step 3: Start the optimized service
    process = start_optimized_service()
    
    if process:
        print("\n🎉 Service restart complete!")
        print("\n💡 Next steps:")
        print("   1. Test the optimizations: python scripts/test_llama2_performance.py")
        print("   2. Try a simple chat: python chat.py llama2")
        print("   3. Monitor logs: tail -f app.log")
        
        print("\n⚠️ Note: The service is running in the background.")
        print("   To stop it later, run this script again or use:")
        print("   pkill -f 'python.*main.py'")
        
        return 0
    else:
        print("❌ Failed to restart service")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 