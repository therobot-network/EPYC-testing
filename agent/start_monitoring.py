#!/usr/bin/env python3
"""
ProcessWatcher Background Monitor

This script starts the ProcessWatcher monitoring agent in the background
and sets up automatic log capture for the current terminal session.

Usage:
    python3 start_monitoring.py --api-url https://your-url.com &
    
After running this, any commands you run in the terminal will automatically
have their output sent to ProcessWatcher.
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import threading
from datetime import datetime
from pathlib import Path

# Add current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitor import ProcessMonitorAgent
from logger_client import ProcessWatcherLogger


class BackgroundMonitor:
    """Background monitoring service that captures terminal activity."""
    
    def __init__(self, api_url, agent_id=None, session_name=None):
        self.api_url = api_url
        self.agent_id = agent_id
        self.session_name = session_name or f"terminal-{os.getpid()}"
        
        # Initialize the system monitoring agent
        self.system_agent = ProcessMonitorAgent(api_url=api_url, agent_id=agent_id)
        
        # Initialize the logger for terminal activity
        self.terminal_logger = ProcessWatcherLogger(
            api_url=api_url,
            agent_id=agent_id,
            app_name=self.session_name,
            buffer_size=50,
            flush_interval=3
        )
        
        # Control flags
        self.running = False
        self.system_monitor_thread = None
        self.log_monitor_thread = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"ProcessWatcher Background Monitor initialized")
        print(f"API URL: {self.api_url}")
        print(f"Agent ID: {self.system_agent.agent_id}")
        print(f"Session: {self.session_name}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the background monitoring."""
        if self.running:
            return
        
        self.running = True
        
        # Start system monitoring in a separate thread
        self.system_monitor_thread = threading.Thread(
            target=self._run_system_monitor,
            daemon=True
        )
        self.system_monitor_thread.start()
        
        # Start terminal log monitoring
        self.log_monitor_thread = threading.Thread(
            target=self._setup_terminal_capture,
            daemon=True
        )
        self.log_monitor_thread.start()
        
        # Log that monitoring started
        self.terminal_logger.info("ProcessWatcher background monitoring started", {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "session": self.session_name,
            "working_directory": os.getcwd(),
            "user": os.getenv('USER', 'unknown')
        })
        
        print(f"Background monitoring started (PID: {os.getpid()})")
        print("Commands run in this terminal will now be logged to ProcessWatcher")
        print("Press Ctrl+C to stop monitoring")
    
    def _run_system_monitor(self):
        """Run the system monitoring agent."""
        try:
            # Register the agent first
            if not self.system_agent.register_agent():
                print("Warning: Failed to register system monitoring agent")
            
            # Start monitoring loop
            while self.running:
                try:
                    self.system_agent.send_data()
                    time.sleep(30)  # Send system data every 30 seconds
                except Exception as e:
                    print(f"System monitoring error: {e}")
                    time.sleep(5)  # Short delay before retry
                    
        except Exception as e:
            print(f"System monitor thread error: {e}")
    
    def _setup_terminal_capture(self):
        """Setup terminal command capture using script command."""
        try:
            # Create a named pipe for capturing terminal output
            pipe_path = f"/tmp/processwatcher_pipe_{os.getpid()}"
            
            # Remove existing pipe if it exists
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
            
            # Create named pipe
            os.mkfifo(pipe_path)
            
            # Setup environment variables for command logging
            self._setup_command_logging()
            
            # Monitor the pipe for output
            self._monitor_pipe(pipe_path)
            
        except Exception as e:
            print(f"Terminal capture setup error: {e}")
    
    def _setup_command_logging(self):
        """Setup environment variables to capture command execution."""
        # Create a wrapper script that logs commands
        wrapper_script = f"/tmp/processwatcher_wrapper_{os.getpid()}.sh"
        
        wrapper_content = f'''#!/bin/bash
# ProcessWatcher command wrapper
ORIGINAL_CMD="$@"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
EXIT_CODE=0

# Log command start
python3 -c "
import sys
sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
from logger_client import ProcessWatcherLogger
logger = ProcessWatcherLogger('{self.api_url}', app_name='{self.session_name}')
logger.info('Command started', {{
    'command': '$ORIGINAL_CMD',
    'start_time': '$START_TIME',
    'working_directory': '$(pwd)',
    'user': '$(whoami)'
}})
logger.shutdown()
"

# Execute the original command and capture output
exec "$@"
EXIT_CODE=$?

# Log command completion
python3 -c "
import sys
sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
from logger_client import ProcessWatcherLogger
logger = ProcessWatcherLogger('{self.api_url}', app_name='{self.session_name}')
logger.info('Command completed', {{
    'command': '$ORIGINAL_CMD',
    'exit_code': $EXIT_CODE,
    'end_time': '$(date '+%Y-%m-%d %H:%M:%S')'
}})
logger.shutdown()
"

exit $EXIT_CODE
'''
        
        with open(wrapper_script, 'w') as f:
            f.write(wrapper_content)
        
        os.chmod(wrapper_script, 0o755)
        
        # Set environment variable to use our wrapper
        print(f"To enable command logging, run:")
        print(f"export PROCESSWATCHER_WRAPPER='{wrapper_script}'")
        print(f"alias run='$PROCESSWATCHER_WRAPPER'")
        print()
        print("Then use 'run <command>' to automatically log commands")
        print("Example: run ls -la")
        print("Example: run python3 my_script.py")
    
    def _monitor_pipe(self, pipe_path):
        """Monitor the named pipe for output."""
        try:
            while self.running:
                try:
                    with open(pipe_path, 'r') as pipe:
                        for line in pipe:
                            if line.strip():
                                self.terminal_logger.info("Terminal output", {
                                    "output": line.strip(),
                                    "timestamp": datetime.now().isoformat()
                                })
                except (BrokenPipeError, FileNotFoundError):
                    if self.running:
                        time.sleep(1)  # Wait before retrying
                except Exception as e:
                    print(f"Pipe monitoring error: {e}")
                    time.sleep(1)
        except Exception as e:
            print(f"Pipe monitor error: {e}")
    
    def stop(self):
        """Stop the background monitoring."""
        if not self.running:
            return
        
        self.running = False
        
        # Log shutdown
        try:
            self.terminal_logger.info("ProcessWatcher background monitoring stopped", {
                "session": self.session_name,
                "shutdown_time": datetime.now().isoformat()
            })
            self.terminal_logger.shutdown()
        except:
            pass
        
        # Clean up temporary files
        try:
            wrapper_script = f"/tmp/processwatcher_wrapper_{os.getpid()}.sh"
            if os.path.exists(wrapper_script):
                os.unlink(wrapper_script)
            
            pipe_path = f"/tmp/processwatcher_pipe_{os.getpid()}"
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
        except:
            pass
        
        print("Background monitoring stopped")
    
    def run_forever(self):
        """Run the monitor until interrupted."""
        self.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='ProcessWatcher Background Monitor')
    parser.add_argument('--api-url', required=True, help='ProcessWatcher API URL')
    parser.add_argument('--agent-id', help='Custom agent ID')
    parser.add_argument('--session-name', help='Custom session name')
    parser.add_argument('--daemon', '-d', action='store_true', help='Run as daemon')
    parser.add_argument('--pid-file', help='PID file location')
    
    args = parser.parse_args()
    
    # Create monitor instance
    monitor = BackgroundMonitor(
        api_url=args.api_url,
        agent_id=args.agent_id,
        session_name=args.session_name
    )
    
    # Write PID file if requested
    if args.pid_file:
        with open(args.pid_file, 'w') as f:
            f.write(str(os.getpid()))
    
    if args.daemon:
        # Fork to background
        if os.fork() > 0:
            sys.exit(0)  # Parent exits
        
        # Child continues as daemon
        os.setsid()
        os.chdir('/')
        os.umask(0)
        
        # Redirect stdout/stderr
        with open('/dev/null', 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
    
    # Run the monitor
    monitor.run_forever()


if __name__ == "__main__":
    main() 