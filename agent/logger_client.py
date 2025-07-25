#!/usr/bin/env python3
"""
ProcessWatcher Logger Client

A lightweight logging client that can be imported into any Python project
to send application logs to the ProcessWatcher system.

Usage:
    from logger_client import ProcessWatcherLogger
    
    # Initialize logger
    logger = ProcessWatcherLogger(api_url="https://your-processwatcher-url.com")
    
    # Use in your application
    logger.info("Application started")
    logger.error("Something went wrong", {"error_code": 500})
"""

import json
import time
import uuid
import socket
import requests
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from queue import Queue, Empty
import atexit
import os


class ProcessWatcherLogger:
    """Lightweight logger client for sending application logs to ProcessWatcher."""
    
    def __init__(self, api_url: str, agent_id: str = None, app_name: str = None, 
                 buffer_size: int = 100, flush_interval: int = 5, async_mode: bool = True):
        """
        Initialize the ProcessWatcher logger client.
        
        Args:
            api_url: URL of the ProcessWatcher API server
            agent_id: Optional agent ID (auto-generated if not provided)
            app_name: Name of your application (defaults to script name)
            buffer_size: Number of logs to buffer before sending
            flush_interval: Seconds between automatic flushes
            async_mode: Whether to send logs asynchronously
        """
        self.api_url = api_url.rstrip('/')
        self.agent_id = agent_id or str(uuid.uuid4())
        self.app_name = app_name or self._get_app_name()
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.async_mode = async_mode
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.timeout = 10
        
        # Log buffer and threading
        self.log_buffer: Queue = Queue(maxsize=buffer_size * 2)
        self.shutdown_event = threading.Event()
        
        # Metadata
        self.hostname = socket.gethostname()
        self.process_id = os.getpid()
        
        # Start background thread for async logging
        if self.async_mode:
            self.worker_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.worker_thread.start()
            
            # Register cleanup on exit
            atexit.register(self.shutdown)
    
    def _get_app_name(self) -> str:
        """Get application name from the main module."""
        import __main__
        if hasattr(__main__, '__file__') and __main__.__file__:
            return os.path.basename(__main__.__file__).replace('.py', '')
        return 'python_app'
    
    def _create_log_entry(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a structured log entry."""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.upper(),
            "message": message,
            "context": context or {},
            "agent_id": self.agent_id,
            "app_name": self.app_name,
            "hostname": self.hostname,
            "process_id": self.process_id,
            "source": "application"
        }
    
    def _send_logs(self, logs: List[Dict[str, Any]]) -> bool:
        """Send logs to the ProcessWatcher API."""
        try:
            payload = {
                "agent_id": self.agent_id,
                "app_name": self.app_name,
                "logs": logs
            }
            
            response = self.session.post(
                f"{self.api_url}/api/v1/logs",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            # Fallback to console logging if API fails
            print(f"ProcessWatcherLogger: Failed to send logs - {e}")
            return False
    
    def _log_worker(self):
        """Background worker thread for processing logs."""
        logs_to_send = []
        last_flush = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                # Try to get a log entry with timeout
                try:
                    log_entry = self.log_buffer.get(timeout=1.0)
                    logs_to_send.append(log_entry)
                    self.log_buffer.task_done()
                except Empty:
                    pass
                
                # Send logs if buffer is full or flush interval reached
                current_time = time.time()
                should_flush = (
                    len(logs_to_send) >= self.buffer_size or
                    (logs_to_send and current_time - last_flush >= self.flush_interval)
                )
                
                if should_flush:
                    if self._send_logs(logs_to_send):
                        logs_to_send.clear()
                        last_flush = current_time
                    else:
                        # Keep logs in buffer if send failed, but limit growth
                        if len(logs_to_send) > self.buffer_size * 2:
                            logs_to_send = logs_to_send[-self.buffer_size:]
                
            except Exception as e:
                print(f"ProcessWatcherLogger worker error: {e}")
        
        # Send remaining logs on shutdown
        if logs_to_send:
            self._send_logs(logs_to_send)
    
    def _log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Internal logging method."""
        log_entry = self._create_log_entry(level, message, context)
        
        if self.async_mode:
            try:
                self.log_buffer.put_nowait(log_entry)
            except:
                # Buffer full, send immediately
                self._send_logs([log_entry])
        else:
            # Synchronous mode - send immediately
            self._send_logs([log_entry])
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log a debug message."""
        self._log("debug", message, context)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log an info message."""
        self._log("info", message, context)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log a warning message."""
        self._log("warning", message, context)
    
    def warn(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Alias for warning."""
        self.warning(message, context)
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log an error message."""
        self._log("error", message, context)
    
    def critical(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log a critical message."""
        self._log("critical", message, context)
    
    def flush(self):
        """Force flush all buffered logs."""
        if self.async_mode:
            # Wait for queue to be processed
            self.log_buffer.join()
        # In sync mode, logs are already sent
    
    def shutdown(self):
        """Gracefully shutdown the logger."""
        if self.async_mode and hasattr(self, 'worker_thread'):
            self.shutdown_event.set()
            self.flush()
            self.worker_thread.join(timeout=5)
        
        self.session.close()


# Convenience functions for quick usage
_default_logger: Optional[ProcessWatcherLogger] = None

def setup_logger(api_url: str, **kwargs) -> ProcessWatcherLogger:
    """Setup a default global logger instance."""
    global _default_logger
    _default_logger = ProcessWatcherLogger(api_url, **kwargs)
    return _default_logger

def get_logger() -> Optional[ProcessWatcherLogger]:
    """Get the default global logger instance."""
    return _default_logger

# Quick logging functions using default logger
def debug(message: str, context: Optional[Dict[str, Any]] = None):
    if _default_logger:
        _default_logger.debug(message, context)

def info(message: str, context: Optional[Dict[str, Any]] = None):
    if _default_logger:
        _default_logger.info(message, context)

def warning(message: str, context: Optional[Dict[str, Any]] = None):
    if _default_logger:
        _default_logger.warning(message, context)

def error(message: str, context: Optional[Dict[str, Any]] = None):
    if _default_logger:
        _default_logger.error(message, context)

def critical(message: str, context: Optional[Dict[str, Any]] = None):
    if _default_logger:
        _default_logger.critical(message, context)


if __name__ == "__main__":
    # Example usage
    import time
    
    # Initialize logger
    logger = ProcessWatcherLogger("http://localhost:8080", app_name="test_app")
    
    # Send some test logs
    logger.info("Application started", {"version": "1.0.0"})
    logger.debug("Debug information", {"user_id": 123})
    logger.warning("This is a warning", {"memory_usage": "85%"})
    logger.error("An error occurred", {"error_code": "E001", "details": "Connection failed"})
    
    # Wait a bit for async processing
    time.sleep(2)
    logger.flush()
    
    print("Test logs sent!")
    logger.shutdown() 