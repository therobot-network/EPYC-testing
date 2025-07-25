#!/usr/bin/env python3
"""
EC2 Process Monitoring Agent

This agent collects system and process information from EC2 instances
and sends it to the ProcessWatcher API server.
"""

import json
import time
import uuid
import socket
import psutil
import requests
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse


class ProcessMonitorAgent:
    """Agent that monitors system processes and sends data to the API server."""
    
    def __init__(self, config: Dict[str, Any] = None, api_url: str = None, agent_id: str = None, instance_id: str = None):
        # Load configuration
        if config:
            self.config = config
        else:
            self.config = {
                "api_url": api_url or "http://localhost:8080",
                "agent_id": agent_id,
                "instance_id": instance_id,
                "collection_interval": 30,
                "process_limit": 50,
                "log_level": "INFO",
                "auto_register": True,
                "retry_attempts": 3,
                "retry_delay": 5
            }
        
        self.api_url = self.config["api_url"].rstrip('/')
        self.agent_id = self.config.get("agent_id") or str(uuid.uuid4())
        self.instance_id = self.config.get("instance_id") or self._get_instance_id()
        self.session = requests.Session()
        self.logger = self._setup_logging()
        
        # Agent metadata
        self.agent_name = f"monitor-{socket.gethostname()}"
        self.version = "1.0.0"
        self.ip_address = self._get_local_ip()
        
        # Configuration settings
        self.collection_interval = self.config.get("collection_interval", 30)
        self.process_limit = self.config.get("process_limit", 50)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_delay = self.config.get("retry_delay", 5)
        
        # Registration status
        self.registered = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper(), logging.INFO)
        
        # Determine log file path
        log_file = None
        if os.path.exists('/var/log/processwatcher'):
            log_file = '/var/log/processwatcher/agent.log'
        elif os.path.exists('/var/log'):
            log_file = '/var/log/ec2-monitor.log'
        
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        return logging.getLogger('ProcessMonitorAgent')
    
    def _get_instance_id(self) -> str:
        """Get EC2 instance ID from metadata service."""
        try:
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/instance-id',
                timeout=2
            )
            if response.status_code == 200:
                return response.text
        except Exception as e:
            self.logger.warning(f"Could not get EC2 instance ID: {e}")
        
        # Fallback to hostname if not on EC2
        return socket.gethostname()
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def register_agent(self) -> bool:
        """Register this agent with the API server."""
        registration_data = {
            "id": self.agent_id,
            "instance_id": self.instance_id,
            "name": self.agent_name,
            "ip_address": self.ip_address,
            "version": self.version
        }
        
        try:
            response = self.session.post(
                f"{self.api_url}/api/v1/agents/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 201:
                self.logger.info(f"Agent {self.agent_id} registered successfully")
                self.registered = True
                return True
            else:
                self.logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            return False
    
    def collect_system_info(self) -> Dict[str, Any]:
        """Collect system-level information."""
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory information
            memory = psutil.virtual_memory()
            
            # Disk information
            disk = psutil.disk_usage('/')
            
            # Load average (Unix-like systems only)
            try:
                load_avg = list(os.getloadavg())
            except (OSError, AttributeError):
                load_avg = [0.0, 0.0, 0.0]
            
            # System uptime
            boot_time = psutil.boot_time()
            uptime = int(time.time() - boot_time)
            
            return {
                "cpu_count": cpu_count,
                "cpu_percent": cpu_percent,
                "memory_total": memory.total,
                "memory_used": memory.used,
                "memory_percent": memory.percent,
                "disk_total": disk.total,
                "disk_used": disk.used,
                "disk_percent": (disk.used / disk.total) * 100,
                "load_average": load_avg,
                "uptime": uptime
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting system info: {e}")
            return {}
    
    def collect_process_info(self, limit: int = None) -> List[Dict[str, Any]]:
        """Collect information about running processes."""
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 
                                           'status', 'create_time', 'username', 'cmdline']):
                try:
                    pinfo = proc.info
                    
                    # Calculate memory in MB
                    memory_mb = pinfo['memory_info'].rss / (1024 * 1024) if pinfo['memory_info'] else 0
                    
                    # Get command line as string
                    cmdline = ' '.join(pinfo['cmdline']) if pinfo['cmdline'] else ''
                    
                    process_data = {
                        "pid": pinfo['pid'],
                        "name": pinfo['name'] or 'Unknown',
                        "cpu_percent": pinfo['cpu_percent'] or 0.0,
                        "memory_mb": round(memory_mb, 2),
                        "status": pinfo['status'] or 'unknown',
                        "create_time": int(pinfo['create_time']) if pinfo['create_time'] else 0,
                        "username": pinfo['username'] or 'unknown',
                        "command_line": cmdline[:500]  # Limit command line length
                    }
                    
                    processes.append(process_data)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # Process may have terminated or we don't have permission
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error collecting process info: {e}")
        
        # Sort by CPU usage and limit results
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        limit = limit or self.process_limit
        return processes[:limit]
    
    def send_data(self) -> bool:
        """Send collected data to the API server with retry logic."""
        if not self.registered:
            if not self.register_agent():
                return False
        
        # Collect all data
        system_info = self.collect_system_info()
        processes = self.collect_process_info()
        
        payload = {
            "agent_id": self.agent_id,
            "instance_id": self.instance_id,
            "system_info": system_info,
            "processes": processes,
            "status": "active"
        }
        
        # Retry logic
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.post(
                    f"{self.api_url}/api/v1/agents/{self.agent_id}/data",
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.logger.debug("Data sent successfully")
                    return True
                else:
                    self.logger.error(f"Failed to send data (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    # If agent not found, try to re-register
                    if response.status_code == 404:
                        self.registered = False
                        break
                    
            except Exception as e:
                self.logger.error(f"Error sending data (attempt {attempt + 1}): {e}")
            
            # Wait before retrying (except on last attempt)
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay)
        
        return False
    
    def run(self, interval: int = None):
        """Run the monitoring agent continuously."""
        interval = interval or self.collection_interval
        
        self.logger.info(f"Starting ProcessMonitor Agent {self.agent_id}")
        self.logger.info(f"Instance ID: {self.instance_id}")
        self.logger.info(f"API URL: {self.api_url}")
        self.logger.info(f"Collection interval: {interval} seconds")
        
        while True:
            try:
                success = self.send_data()
                if success:
                    self.logger.debug(f"Data collection cycle completed")
                else:
                    self.logger.warning("Data collection cycle failed")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                self.logger.info("Shutting down agent...")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                time.sleep(interval)


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)


def main():
    """Main entry point for the agent."""
    parser = argparse.ArgumentParser(description='EC2 Process Monitoring Agent')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--api-url', help='API server URL (overrides config)')
    parser.add_argument('--agent-id', help='Agent ID (overrides config)')
    parser.add_argument('--instance-id', help='Instance ID (overrides config)')
    parser.add_argument('--interval', type=int, help='Collection interval in seconds (overrides config)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override config with command line arguments
    if args.api_url:
        config['api_url'] = args.api_url
    if args.agent_id:
        config['agent_id'] = args.agent_id
    if args.instance_id:
        config['instance_id'] = args.instance_id
    if args.interval:
        config['collection_interval'] = args.interval
    if args.verbose:
        config['log_level'] = 'DEBUG'
    
    # Validate required configuration
    if not config.get('api_url') and not args.api_url:
        print("Error: API URL must be provided via config file or --api-url argument")
        sys.exit(1)
    
    # Create and run agent
    if config:
        agent = ProcessMonitorAgent(config=config)
    else:
        agent = ProcessMonitorAgent(
            api_url=args.api_url,
            agent_id=args.agent_id,
            instance_id=args.instance_id
        )
    
    agent.run(interval=args.interval)


if __name__ == "__main__":
    main() 