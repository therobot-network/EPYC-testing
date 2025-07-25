#!/usr/bin/env python3
"""
Example: How to integrate ProcessWatcher logging into your existing Python project

This example shows different ways to use the ProcessWatcher logger client
in your applications to send logs to the ProcessWatcher dashboard.
"""

import time
import random
from logger_client import ProcessWatcherLogger, setup_logger, info, error, warning

def example_basic_usage():
    """Example 1: Basic usage with individual logger instance"""
    print("=== Example 1: Basic Usage ===")
    
    # Initialize logger with your ProcessWatcher API URL
    logger = ProcessWatcherLogger(
        api_url="http://localhost:8080",  # Replace with your ProcessWatcher URL
        app_name="my_web_app",
        agent_id="web-server-001"  # Optional: specify agent ID
    )
    
    # Use the logger in your application
    logger.info("Application started", {"version": "1.2.3", "port": 8080})
    logger.debug("Database connection established", {"host": "localhost", "db": "myapp"})
    logger.warning("High memory usage detected", {"memory_percent": 85})
    logger.error("Failed to process user request", {
        "user_id": 12345,
        "error_code": "AUTH_FAILED",
        "endpoint": "/api/users/profile"
    })
    
    # Flush logs and cleanup
    logger.flush()
    logger.shutdown()
    print("Basic usage example completed")

def example_global_logger():
    """Example 2: Using global logger functions"""
    print("\n=== Example 2: Global Logger ===")
    
    # Setup global logger once at the start of your application
    setup_logger(
        api_url="http://localhost:8080",
        app_name="background_worker",
        buffer_size=50,  # Buffer 50 logs before sending
        flush_interval=10  # Send logs every 10 seconds
    )
    
    # Now you can use the convenience functions anywhere in your code
    info("Worker process started", {"worker_id": "worker-001"})
    
    # Simulate some work with logging
    for i in range(5):
        info(f"Processing job {i+1}", {"job_id": f"job_{i+1}", "progress": (i+1)/5 * 100})
        time.sleep(0.5)
    
    warning("Job queue is getting full", {"queue_size": 95, "max_size": 100})
    info("All jobs completed successfully")
    print("Global logger example completed")

def example_web_application():
    """Example 3: Integration with a web application"""
    print("\n=== Example 3: Web Application Integration ===")
    
    # This would typically be in your main application file
    app_logger = ProcessWatcherLogger(
        api_url="http://localhost:8080",
        app_name="flask_api",
        async_mode=True  # Use async mode for better performance
    )
    
    def simulate_web_request(user_id, endpoint):
        """Simulate handling a web request with logging"""
        request_id = f"req_{random.randint(1000, 9999)}"
        
        app_logger.info("Request started", {
            "request_id": request_id,
            "user_id": user_id,
            "endpoint": endpoint,
            "method": "POST"
        })
        
        # Simulate some processing
        time.sleep(0.1)
        
        if random.random() < 0.8:  # 80% success rate
            app_logger.info("Request completed successfully", {
                "request_id": request_id,
                "status_code": 200,
                "response_time_ms": 150
            })
        else:
            app_logger.error("Request failed", {
                "request_id": request_id,
                "status_code": 500,
                "error": "Database connection timeout"
            })
    
    # Simulate multiple requests
    endpoints = ["/api/users", "/api/orders", "/api/products"]
    for i in range(10):
        simulate_web_request(
            user_id=random.randint(1, 1000),
            endpoint=random.choice(endpoints)
        )
    
    app_logger.flush()
    app_logger.shutdown()
    print("Web application example completed")

def example_error_handling():
    """Example 4: Error handling and context logging"""
    print("\n=== Example 4: Error Handling ===")
    
    logger = ProcessWatcherLogger(
        api_url="http://localhost:8080",
        app_name="data_processor"
    )
    
    def process_data(data_id):
        """Simulate data processing with comprehensive logging"""
        try:
            logger.info("Starting data processing", {"data_id": data_id})
            
            # Simulate some processing steps
            logger.debug("Loading data from database", {"data_id": data_id})
            time.sleep(0.1)
            
            logger.debug("Validating data format", {"data_id": data_id})
            
            # Simulate occasional errors
            if random.random() < 0.2:  # 20% error rate
                raise ValueError(f"Invalid data format for ID {data_id}")
            
            logger.debug("Transforming data", {"data_id": data_id})
            time.sleep(0.1)
            
            logger.info("Data processing completed", {
                "data_id": data_id,
                "processing_time_ms": 200,
                "records_processed": random.randint(100, 1000)
            })
            
        except Exception as e:
            logger.error("Data processing failed", {
                "data_id": data_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "stack_trace": str(e.__traceback__)
            })
            raise
    
    # Process multiple data items
    for data_id in range(1, 11):
        try:
            process_data(f"data_{data_id}")
        except Exception:
            pass  # Continue with next item
    
    logger.flush()
    logger.shutdown()
    print("Error handling example completed")

def example_integration_with_existing_logging():
    """Example 5: Integration with Python's standard logging"""
    print("\n=== Example 5: Integration with Standard Logging ===")
    
    import logging
    
    # Setup ProcessWatcher logger
    pw_logger = ProcessWatcherLogger(
        api_url="http://localhost:8080",
        app_name="hybrid_logging_app"
    )
    
    # Setup standard Python logging
    logging.basicConfig(level=logging.INFO)
    std_logger = logging.getLogger(__name__)
    
    class ProcessWatcherHandler(logging.Handler):
        """Custom logging handler to send logs to ProcessWatcher"""
        def __init__(self, pw_logger):
            super().__init__()
            self.pw_logger = pw_logger
        
        def emit(self, record):
            level_map = {
                logging.DEBUG: 'debug',
                logging.INFO: 'info',
                logging.WARNING: 'warning',
                logging.ERROR: 'error',
                logging.CRITICAL: 'critical'
            }
            
            level = level_map.get(record.levelno, 'info')
            message = record.getMessage()
            context = {
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Send to ProcessWatcher
            self.pw_logger._log(level, message, context)
    
    # Add ProcessWatcher handler to standard logger
    pw_handler = ProcessWatcherHandler(pw_logger)
    std_logger.addHandler(pw_handler)
    
    # Now standard logging calls will also go to ProcessWatcher
    std_logger.info("This goes to both console and ProcessWatcher")
    std_logger.warning("Warning message with dual logging")
    std_logger.error("Error message visible in both places")
    
    pw_logger.flush()
    pw_logger.shutdown()
    print("Hybrid logging example completed")

if __name__ == "__main__":
    print("ProcessWatcher Logger Integration Examples")
    print("=" * 50)
    print("Make sure your ProcessWatcher server is running at http://localhost:8080")
    print("You can change the API URL in each example to match your setup.")
    print()
    
    try:
        example_basic_usage()
        example_global_logger()
        example_web_application()
        example_error_handling()
        example_integration_with_existing_logging()
        
        print("\n" + "=" * 50)
        print("All examples completed! Check your ProcessWatcher dashboard to see the logs.")
        print("You can view logs at: http://localhost:8080 (or your ProcessWatcher URL)")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure your ProcessWatcher server is running and accessible.") 