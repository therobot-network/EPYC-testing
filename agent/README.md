# ProcessWatcher Agent

This directory contains the ProcessWatcher monitoring agent and logging client that can be deployed on EC2 instances or integrated into Python applications.

## Files

- **`monitor.py`** - Full monitoring agent that collects system metrics and process information
- **`logger_client.py`** - Lightweight logging client for sending application logs to ProcessWatcher
- **`example_usage.py`** - Examples showing how to integrate the logger into your Python projects
- **`requirements.txt`** - Python dependencies

## Quick Start

### 1. System Monitoring Agent

Deploy the full monitoring agent on EC2 instances to collect system metrics:

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the monitoring agent
python3 monitor.py --api-url https://your-processwatcher-url.com --verbose
```

### 2. Application Logging (New!)

Integrate ProcessWatcher logging into your existing Python applications:

```python
from logger_client import ProcessWatcherLogger

# Initialize logger
logger = ProcessWatcherLogger(api_url="https://your-processwatcher-url.com")

# Use in your application
logger.info("Application started", {"version": "1.0.0"})
logger.error("Something went wrong", {"error_code": 500})
```

## Integration into Existing Projects

### Method 1: Copy Files

1. Copy `logger_client.py` into your existing Python project
2. Import and use the logger:

```python
from logger_client import ProcessWatcherLogger

logger = ProcessWatcherLogger("https://your-processwatcher-url.com")
logger.info("Hello from my app!")
```

### Method 2: Git Submodule (Recommended)

1. Add ProcessWatcher as a submodule to your project:
```bash
git submodule add https://github.com/YourUsername/ProcessWatcher.git processwatcher
```

2. Import from the agent directory:
```python
from processwatcher.web.agent.logger_client import ProcessWatcherLogger

logger = ProcessWatcherLogger("https://your-processwatcher-url.com")
logger.info("Logging from my main project!")
```

### Method 3: Global Logger

For simpler integration, use the global logger functions:

```python
from processwatcher.web.agent.logger_client import setup_logger, info, error

# Setup once at app startup
setup_logger("https://your-processwatcher-url.com", app_name="my_app")

# Use anywhere in your code
info("User logged in", {"user_id": 123})
error("Database connection failed")
```

## Features

### Logger Client Features

- **Asynchronous logging** - Non-blocking log submission
- **Automatic buffering** - Batches logs for efficient transmission
- **Error resilience** - Falls back to console logging if API is unavailable
- **Rich context** - Support for structured logging with custom fields
- **Multiple log levels** - Debug, Info, Warning, Error, Critical
- **Easy integration** - Works with existing Python logging

### Monitoring Agent Features

- **System metrics** - CPU, memory, disk usage
- **Process information** - Running processes with resource usage
- **Automatic registration** - Self-registers with ProcessWatcher API
- **Configurable intervals** - Customizable data collection frequency

## Configuration

### Logger Client Options

```python
logger = ProcessWatcherLogger(
    api_url="https://your-url.com",     # Required: ProcessWatcher API URL
    agent_id="my-app-001",              # Optional: Custom agent ID
    app_name="my_application",          # Optional: Application name
    buffer_size=100,                    # Optional: Log buffer size
    flush_interval=5,                   # Optional: Seconds between flushes
    async_mode=True                     # Optional: Enable async logging
)
```

### Monitoring Agent Options

```bash
python3 monitor.py \
    --api-url https://your-url.com \    # Required: ProcessWatcher API URL
    --agent-id custom-agent-id \        # Optional: Custom agent ID
    --instance-id i-1234567890 \        # Optional: EC2 instance ID
    --interval 30 \                     # Optional: Collection interval (seconds)
    --verbose                           # Optional: Enable debug logging
```

## API Endpoints

The logger client sends logs to these ProcessWatcher API endpoints:

- `POST /api/v1/logs` - Submit application logs
- `GET /api/v1/logs` - Retrieve logs (authenticated)
- `GET /api/v1/agents/:id/logs` - Get logs for specific agent
- `DELETE /api/v1/logs` - Clear all logs (authenticated)

## Examples

See `example_usage.py` for comprehensive examples including:

1. **Basic Usage** - Simple logging setup
2. **Global Logger** - Using convenience functions
3. **Web Application** - Integration with Flask/Django
4. **Error Handling** - Comprehensive error logging
5. **Hybrid Logging** - Integration with Python's standard logging

## Log Levels

- **DEBUG** - Detailed diagnostic information
- **INFO** - General information about application flow
- **WARN** - Warning messages for potential issues
- **ERROR** - Error messages for handled exceptions
- **CRITICAL** - Critical errors that may cause application failure

## Viewing Logs

Once integrated, your application logs will appear in the ProcessWatcher dashboard:

1. Navigate to your ProcessWatcher URL
2. Login to the dashboard
3. View logs in the "Application Logs" section
4. Filter by agent, application, or log level

## Troubleshooting

### Common Issues

1. **Connection Failed**: Ensure ProcessWatcher server is running and accessible
2. **Authentication Error**: Check if your ProcessWatcher instance requires authentication
3. **Logs Not Appearing**: Verify the API URL and check server logs

### Debug Mode

Enable verbose logging to troubleshoot issues:

```python
logger = ProcessWatcherLogger(
    api_url="https://your-url.com",
    app_name="debug_app"
)

# Logger will print debug information to console if API calls fail
```

## Requirements

- Python 3.6+
- `requests` library
- `psutil` library (for monitoring agent only)

Install with:
```bash
pip3 install -r requirements.txt
``` 