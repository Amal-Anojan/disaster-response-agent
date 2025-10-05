import logging
import traceback
import sys
from typing import Any, Dict, Optional, Callable, Union
from datetime import datetime
from functools import wraps
import asyncio
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self):
        """Initialize error handler"""
        self.error_counts = {}
        self.error_history = []
        self.max_history = 1000
        
        # Error severity levels
        self.severity_levels = {
            'CRITICAL': 50,
            'HIGH': 40,
            'MEDIUM': 30,
            'LOW': 20,
            'INFO': 10
        }
    
    def handle_error(self, 
                    error: Exception, 
                    context: Optional[Dict[str, Any]] = None,
                    severity: str = 'MEDIUM',
                    user_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle and log error with context
        
        Args:
            error: The exception that occurred
            context: Additional context information
            severity: Error severity level
            user_message: User-friendly error message
            
        Returns:
            Error information dictionary
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'user_message': user_message,
            'traceback': traceback.format_exc()
        }
        
        # Log error based on severity
        log_level = getattr(logging, severity, logging.ERROR)
        logger.log(log_level, f"Error handled: {error_info['error_type']} - {error_info['error_message']}")
        
        # Update error statistics
        error_type = error_info['error_type']
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to error history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        return error_info
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            'total_errors': len(self.error_history),
            'error_counts_by_type': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.error_counts.clear()

# Global error handler instance
_error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    return _error_handler

# Decorator for error handling
def handle_errors(severity: str = 'MEDIUM', 
                 user_message: Optional[str] = None,
                 reraise: bool = True,
                 return_value: Any = None):
    """
    Decorator to handle errors in functions
    
    Args:
        severity: Error severity level
        user_message: User-friendly error message
        reraise: Whether to reraise the exception
        return_value: Value to return if error occurs and reraise=False
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Limit length
                    'kwargs': str(kwargs)[:200]
                }
                
                error_info = get_error_handler().handle_error(
                    e, context, severity, user_message
                )
                
                if reraise:
                    raise
                else:
                    return return_value
        return wrapper
    return decorator

def async_handle_errors(severity: str = 'MEDIUM',
                       user_message: Optional[str] = None,
                       reraise: bool = True,
                       return_value: Any = None):
    """
    Decorator to handle errors in async functions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                }
                
                error_info = get_error_handler().handle_error(
                    e, context, severity, user_message
                )
                
                if reraise:
                    raise
                else:
                    return return_value
        return wrapper
    return decorator

# Context manager for error handling
@contextmanager
def error_context(context_name: str, 
                 additional_context: Optional[Dict[str, Any]] = None,
                 severity: str = 'MEDIUM'):
    """
    Context manager for error handling
    
    Args:
        context_name: Name of the context
        additional_context: Additional context information
        severity: Error severity level
    """
    try:
        yield
    except Exception as e:
        context = {
            'context_name': context_name,
            **(additional_context or {})
        }
        
        get_error_handler().handle_error(e, context, severity)
        raise

# Specific error classes for the disaster response system
class DisasterResponseError(Exception):
    """Base exception for disaster response system"""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}

class VisionAnalysisError(DisasterResponseError):
    """Exception for vision analysis errors"""
    pass

class LLMServiceError(DisasterResponseError):
    """Exception for LLM service errors"""
    pass

class ResourceAllocationError(DisasterResponseError):
    """Exception for resource allocation errors"""
    pass

class DatabaseError(DisasterResponseError):
    """Exception for database errors"""
    pass

class APIError(DisasterResponseError):
    """Exception for API errors"""
    pass

class ProcessingError(DisasterResponseError):
    """Exception for processing pipeline errors"""
    pass

# Error recovery strategies
class ErrorRecovery:
    """Error recovery and retry strategies"""
    
    def __init__(self):
        """Initialize error recovery"""
        self.retry_strategies = {
            'exponential_backoff': self._exponential_backoff_retry,
            'linear_backoff': self._linear_backoff_retry,
            'immediate_retry': self._immediate_retry
        }
    
    def retry_with_strategy(self, 
                          func: Callable,
                          strategy: str = 'exponential_backoff',
                          max_retries: int = 3,
                          base_delay: float = 1.0,
                          max_delay: float = 60.0,
                          exceptions: tuple = (Exception,)) -> Any:
        """
        Retry function with specified strategy
        
        Args:
            func: Function to retry
            strategy: Retry strategy name
            max_retries: Maximum number of retries
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            exceptions: Exceptions to catch and retry on
            
        Returns:
            Function result or raises last exception
        """
        if strategy not in self.retry_strategies:
            raise ValueError(f"Unknown retry strategy: {strategy}")
        
        return self.retry_strategies[strategy](
            func, max_retries, base_delay, max_delay, exceptions
        )
    
    def _exponential_backoff_retry(self, func, max_retries, base_delay, max_delay, exceptions):
        """Exponential backoff retry strategy"""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    break
                
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                time.sleep(delay)
        
        raise last_exception
    
    def _linear_backoff_retry(self, func, max_retries, base_delay, max_delay, exceptions):
        """Linear backoff retry strategy"""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    break
                
                delay = min(base_delay * (attempt + 1), max_delay)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                time.sleep(delay)
        
        raise last_exception
    
    def _immediate_retry(self, func, max_retries, base_delay, max_delay, exceptions):
        """Immediate retry strategy (no delay)"""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt == max_retries:
                    break
                
                logger.warning(f"Attempt {attempt + 1} failed, retrying immediately: {e}")
        
        raise last_exception

# Global error recovery instance
_error_recovery = ErrorRecovery()

def get_error_recovery() -> ErrorRecovery:
    """Get global error recovery instance"""
    return _error_recovery

# Retry decorator
def retry(strategy: str = 'exponential_backoff',
         max_retries: int = 3,
         base_delay: float = 1.0,
         max_delay: float = 60.0,
         exceptions: tuple = (Exception,)):
    """
    Decorator to add retry logic to functions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            def retry_func():
                return func(*args, **kwargs)
            
            return get_error_recovery().retry_with_strategy(
                retry_func, strategy, max_retries, base_delay, max_delay, exceptions
            )
        return wrapper
    return decorator

def async_retry(strategy: str = 'exponential_backoff',
               max_retries: int = 3,
               base_delay: float = 1.0,
               max_delay: float = 60.0,
               exceptions: tuple = (Exception,)):
    """
    Decorator to add retry logic to async functions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    if strategy == 'exponential_backoff':
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    elif strategy == 'linear_backoff':
                        delay = min(base_delay * (attempt + 1), max_delay)
                    else:  # immediate_retry
                        delay = 0
                    
                    logger.warning(f"Async attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    
                    if delay > 0:
                        await asyncio.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator

# Error reporting and monitoring
class ErrorMonitor:
    """Error monitoring and alerting"""
    
    def __init__(self):
        """Initialize error monitor"""
        self.alert_thresholds = {
            'CRITICAL': 1,  # Alert immediately
            'HIGH': 5,      # Alert after 5 errors
            'MEDIUM': 10,   # Alert after 10 errors
            'LOW': 50       # Alert after 50 errors
        }
        
        self.error_counts_by_severity = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0
        }
    
    def record_error(self, severity: str):
        """Record error occurrence"""
        if severity in self.error_counts_by_severity:
            self.error_counts_by_severity[severity] += 1
            
            # Check if alert threshold reached
            if self.error_counts_by_severity[severity] >= self.alert_thresholds[severity]:
                self._trigger_alert(severity, self.error_counts_by_severity[severity])
    
    def _trigger_alert(self, severity: str, count: int):
        """Trigger error alert"""
        logger.error(f"ERROR ALERT: {severity} error threshold reached ({count} errors)")
        
        # In a real system, this would send notifications via:
        # - Email alerts
        # - Slack/Teams notifications
        # - SMS for critical errors
        # - Dashboard alerts
        
        # Reset counter after alert
        self.error_counts_by_severity[severity] = 0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for monitoring"""
        return {
            'error_counts': self.error_counts_by_severity.copy(),
            'alert_thresholds': self.alert_thresholds.copy(),
            'timestamp': datetime.now().isoformat()
        }

# Global error monitor
_error_monitor = ErrorMonitor()

def get_error_monitor() -> ErrorMonitor:
    """Get global error monitor instance"""
    return _error_monitor

# Health check for error handling system
def error_handler_health_check() -> Dict[str, Any]:
    """Check health of error handling system"""
    try:
        error_handler = get_error_handler()
        error_recovery = get_error_recovery()
        error_monitor = get_error_monitor()
        
        return {
            'status': 'healthy',
            'error_handler': {
                'total_errors': len(error_handler.error_history),
                'error_types': len(error_handler.error_counts)
            },
            'error_recovery': {
                'strategies_available': len(error_recovery.retry_strategies)
            },
            'error_monitor': error_monitor.get_error_summary(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Usage examples and testing
def test_error_handling():
    """Test error handling functionality"""
    print("Testing Error Handling...")
    
    # Test basic error handling
    @handle_errors(severity='HIGH', user_message='Test function failed')
    def failing_function():
        raise ValueError("This is a test error")
    
    try:
        failing_function()
    except ValueError:
        print("✓ Error handling decorator works")
    
    # Test retry functionality
    attempt_count = 0
    
    @retry(max_retries=3, base_delay=0.1)
    def sometimes_failing_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"Attempt {attempt_count} failed")
        return "Success!"
    
    try:
        result = sometimes_failing_function()
        print(f"✓ Retry decorator works: {result}")
    except Exception as e:
        print(f"✗ Retry failed: {e}")
    
    # Test error context manager
    try:
        with error_context('test_context', {'test_data': 'value'}):
            raise RuntimeError("Context manager test")
    except RuntimeError:
        print("✓ Error context manager works")
    
    # Test error statistics
    error_handler = get_error_handler()
    stats = error_handler.get_error_stats()
    print(f"✓ Error statistics: {stats['total_errors']} errors recorded")
    
    # Test health check
    health = error_handler_health_check()
    print(f"✓ Health check status: {health['status']}")
    
    print("Error handling tests completed!")

if __name__ == "__main__":
    import time
    test_error_handling()