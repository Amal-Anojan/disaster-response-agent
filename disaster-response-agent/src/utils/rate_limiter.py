import asyncio
import time
import logging
from typing import Dict, Any, Callable, Optional, List
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, rate_limit: float, burst_capacity: int):
        """
        Initialize token bucket rate limiter
        
        Args:
            rate_limit: Tokens per second
            burst_capacity: Maximum tokens in bucket
        """
        self.rate_limit = rate_limit
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def is_allowed(self, tokens_requested: int = 1) -> bool:
        """Check if request is allowed and consume tokens"""
        with self.lock:
            now = time.time()
            
            # Refill tokens based on time elapsed
            time_elapsed = now - self.last_refill
            tokens_to_add = time_elapsed * self.rate_limit
            self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if enough tokens available
            if self.tokens >= tokens_requested:
                self.tokens -= tokens_requested
                return True
            else:
                return False
    
    def time_to_next_token(self) -> float:
        """Get time until next token is available"""
        with self.lock:
            if self.tokens >= 1:
                return 0.0
            else:
                return (1 - self.tokens) / self.rate_limit

class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize sliding window rate limiter
        
        Args:
            max_requests: Maximum requests in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            
            # Check if under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            else:
                return False
    
    def get_reset_time(self) -> Optional[float]:
        """Get time when window resets"""
        with self.lock:
            if not self.requests:
                return None
            return self.requests[0] + self.window_seconds

class RateLimitManager:
    """Centralized rate limit management"""
    
    def __init__(self):
        """Initialize rate limit manager"""
        self.limiters: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
        # Default rate limits for different services
        self.default_limits = {
            'api_general': {'type': 'token_bucket', 'rate': 10, 'burst': 20},
            'api_vision': {'type': 'token_bucket', 'rate': 2, 'burst': 5},
            'api_llm': {'type': 'token_bucket', 'rate': 1, 'burst': 3},
            'user_requests': {'type': 'sliding_window', 'max_requests': 100, 'window': 3600},
            'incident_processing': {'type': 'token_bucket', 'rate': 5, 'burst': 10},
            'database_write': {'type': 'token_bucket', 'rate': 20, 'burst': 50}
        }
    
    def create_limiter(self, name: str, limiter_config: Dict[str, Any]):
        """Create a rate limiter with given configuration"""
        with self.lock:
            if limiter_config['type'] == 'token_bucket':
                self.limiters[name] = TokenBucketRateLimiter(
                    limiter_config['rate'],
                    limiter_config['burst']
                )
            elif limiter_config['type'] == 'sliding_window':
                self.limiters[name] = SlidingWindowRateLimiter(
                    limiter_config['max_requests'],
                    limiter_config['window']
                )
            else:
                raise ValueError(f"Unknown limiter type: {limiter_config['type']}")
    
    def get_limiter(self, name: str) -> Any:
        """Get or create rate limiter"""
        if name not in self.limiters:
            # Create with default configuration if available
            if name in self.default_limits:
                self.create_limiter(name, self.default_limits[name])
            else:
                # Create default token bucket limiter
                self.create_limiter(name, {'type': 'token_bucket', 'rate': 5, 'burst': 10})
        
        return self.limiters[name]
    
    def is_allowed(self, limiter_name: str, tokens: int = 1) -> bool:
        """Check if request is allowed by specific limiter"""
        limiter = self.get_limiter(limiter_name)
        
        if isinstance(limiter, TokenBucketRateLimiter):
            return limiter.is_allowed(tokens)
        elif isinstance(limiter, SlidingWindowRateLimiter):
            return limiter.is_allowed()
        else:
            return True
    
    def wait_for_capacity(self, limiter_name: str, tokens: int = 1, timeout: float = 30) -> bool:
        """Wait until capacity is available or timeout"""
        limiter = self.get_limiter(limiter_name)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_allowed(limiter_name, tokens):
                return True
            
            # Calculate sleep time
            if isinstance(limiter, TokenBucketRateLimiter):
                sleep_time = min(limiter.time_to_next_token(), 1.0)
            else:
                sleep_time = 1.0
            
            time.sleep(sleep_time)
        
        return False
    
    def get_limiter_status(self, limiter_name: str) -> Dict[str, Any]:
        """Get status information for a limiter"""
        if limiter_name not in self.limiters:
            return {'exists': False}
        
        limiter = self.limiters[limiter_name]
        status = {'exists': True, 'type': type(limiter).__name__}
        
        if isinstance(limiter, TokenBucketRateLimiter):
            with limiter.lock:
                status.update({
                    'rate_limit': limiter.rate_limit,
                    'burst_capacity': limiter.burst_capacity,
                    'current_tokens': limiter.tokens,
                    'time_to_next_token': limiter.time_to_next_token()
                })
        elif isinstance(limiter, SlidingWindowRateLimiter):
            with limiter.lock:
                status.update({
                    'max_requests': limiter.max_requests,
                    'window_seconds': limiter.window_seconds,
                    'current_requests': len(limiter.requests),
                    'reset_time': limiter.get_reset_time()
                })
        
        return status
    
    def get_all_limiters_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all rate limiters"""
        return {name: self.get_limiter_status(name) for name in self.limiters.keys()}

# Global rate limit manager instance
_rate_limit_manager = None

def get_rate_limit_manager() -> RateLimitManager:
    """Get global rate limit manager instance"""
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
    return _rate_limit_manager

# Decorators for rate limiting
def rate_limit(limiter_name: str, tokens: int = 1):
    """Decorator to apply rate limiting to functions"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            manager = get_rate_limit_manager()
            
            if not manager.is_allowed(limiter_name, tokens):
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {limiter_name}",
                    limiter_name=limiter_name,
                    retry_after=manager.get_limiter(limiter_name).time_to_next_token() 
                    if isinstance(manager.get_limiter(limiter_name), TokenBucketRateLimiter) else 60
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def async_rate_limit(limiter_name: str, tokens: int = 1):
    """Decorator to apply rate limiting to async functions"""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            manager = get_rate_limit_manager()
            
            if not manager.is_allowed(limiter_name, tokens):
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {limiter_name}",
                    limiter_name=limiter_name,
                    retry_after=manager.get_limiter(limiter_name).time_to_next_token()
                    if isinstance(manager.get_limiter(limiter_name), TokenBucketRateLimiter) else 60
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limit_with_wait(limiter_name: str, tokens: int = 1, timeout: float = 30):
    """Decorator that waits for capacity instead of failing immediately"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            manager = get_rate_limit_manager()
            
            if not manager.wait_for_capacity(limiter_name, tokens, timeout):
                raise RateLimitExceeded(
                    f"Rate limit timeout for {limiter_name}",
                    limiter_name=limiter_name,
                    retry_after=60
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Custom exceptions
class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, message: str, limiter_name: str, retry_after: float = 60):
        super().__init__(message)
        self.limiter_name = limiter_name
        self.retry_after = retry_after

# Context manager for rate limiting
class RateLimitContext:
    """Context manager for rate limiting"""
    
    def __init__(self, limiter_name: str, tokens: int = 1, wait: bool = False, timeout: float = 30):
        self.limiter_name = limiter_name
        self.tokens = tokens
        self.wait = wait
        self.timeout = timeout
        self.manager = get_rate_limit_manager()
    
    def __enter__(self):
        if self.wait:
            if not self.manager.wait_for_capacity(self.limiter_name, self.tokens, self.timeout):
                raise RateLimitExceeded(
                    f"Rate limit timeout for {self.limiter_name}",
                    limiter_name=self.limiter_name,
                    retry_after=60
                )
        else:
            if not self.manager.is_allowed(self.limiter_name, self.tokens):
                limiter = self.manager.get_limiter(self.limiter_name)
                retry_after = (limiter.time_to_next_token() 
                             if isinstance(limiter, TokenBucketRateLimiter) else 60)
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {self.limiter_name}",
                    limiter_name=self.limiter_name,
                    retry_after=retry_after
                )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# API-specific rate limiters
class APIRateLimiter:
    """Specialized rate limiter for API endpoints"""
    
    def __init__(self):
        self.manager = get_rate_limit_manager()
        
        # Create API-specific limiters
        api_limits = {
            'vision_api': {'type': 'token_bucket', 'rate': 2, 'burst': 5},
            'llm_api': {'type': 'token_bucket', 'rate': 1, 'burst': 3},
            'general_api': {'type': 'token_bucket', 'rate': 10, 'burst': 20},
            'user_session': {'type': 'sliding_window', 'max_requests': 100, 'window': 3600}
        }
        
        for name, config in api_limits.items():
            self.manager.create_limiter(name, config)
    
    def check_api_limit(self, api_type: str, user_id: str = None) -> bool:
        """Check API rate limit"""
        # Check general API limit
        if not self.manager.is_allowed('general_api'):
            return False
        
        # Check specific API limit
        if not self.manager.is_allowed(api_type):
            return False
        
        # Check user session limit if user_id provided
        if user_id:
            user_limiter = f"user_{user_id}"
            if not self.manager.is_allowed(user_limiter):
                return False
        
        return True
    
    def get_retry_after(self, api_type: str) -> float:
        """Get retry-after time for API"""
        limiter = self.manager.get_limiter(api_type)
        if isinstance(limiter, TokenBucketRateLimiter):
            return limiter.time_to_next_token()
        elif isinstance(limiter, SlidingWindowRateLimiter):
            reset_time = limiter.get_reset_time()
            return max(0, reset_time - time.time()) if reset_time else 60
        else:
            return 60

# Health monitoring for rate limiters
class RateLimiterHealth:
    """Health monitoring for rate limiters"""
    
    def __init__(self):
        self.manager = get_rate_limit_manager()
        self.health_history = defaultdict(list)
        self.max_history_size = 100
    
    def record_usage(self, limiter_name: str, allowed: bool):
        """Record rate limiter usage"""
        timestamp = time.time()
        self.health_history[limiter_name].append({
            'timestamp': timestamp,
            'allowed': allowed
        })
        
        # Keep only recent history
        if len(self.health_history[limiter_name]) > self.max_history_size:
            self.health_history[limiter_name] = self.health_history[limiter_name][-self.max_history_size:]
    
    def get_health_metrics(self, limiter_name: str, window_minutes: int = 5) -> Dict[str, Any]:
        """Get health metrics for a rate limiter"""
        if limiter_name not in self.health_history:
            return {
                'limiter_name': limiter_name,
                'health_score': 1.0,
                'total_requests': 0,
                'allowed_requests': 0,
                'denied_requests': 0,
                'allow_rate': 0.0
            }
        
        now = time.time()
        window_start = now - (window_minutes * 60)
        
        # Filter history to window
        recent_history = [
            record for record in self.health_history[limiter_name]
            if record['timestamp'] >= window_start
        ]
        
        if not recent_history:
            return {
                'limiter_name': limiter_name,
                'health_score': 1.0,
                'total_requests': 0,
                'allowed_requests': 0,
                'denied_requests': 0,
                'allow_rate': 0.0
            }
        
        total_requests = len(recent_history)
        allowed_requests = sum(1 for r in recent_history if r['allowed'])
        denied_requests = total_requests - allowed_requests
        allow_rate = allowed_requests / total_requests if total_requests > 0 else 0.0
        
        # Calculate health score (1.0 = healthy, 0.0 = unhealthy)
        # High denial rate indicates potential issues
        health_score = allow_rate
        
        return {
            'limiter_name': limiter_name,
            'health_score': health_score,
            'total_requests': total_requests,
            'allowed_requests': allowed_requests,
            'denied_requests': denied_requests,
            'allow_rate': allow_rate,
            'window_minutes': window_minutes
        }
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health of all rate limiters"""
        limiter_healths = {}
        overall_scores = []
        
        for limiter_name in self.health_history.keys():
            health = self.get_health_metrics(limiter_name)
            limiter_healths[limiter_name] = health
            overall_scores.append(health['health_score'])
        
        overall_health_score = sum(overall_scores) / len(overall_scores) if overall_scores else 1.0
        
        return {
            'overall_health_score': overall_health_score,
            'limiter_count': len(limiter_healths),
            'limiter_healths': limiter_healths,
            'timestamp': datetime.now().isoformat()
        }

# Global health monitor
_health_monitor = None

def get_health_monitor() -> RateLimiterHealth:
    """Get global health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = RateLimiterHealth()
    return _health_monitor

# Usage examples and testing
def test_rate_limiters():
    """Test rate limiting functionality"""
    print("Testing Rate Limiters...")
    
    # Test token bucket limiter
    token_limiter = TokenBucketRateLimiter(rate_limit=2.0, burst_capacity=5)
    
    print("Token Bucket Tests:")
    for i in range(8):
        allowed = token_limiter.is_allowed()
        print(f"Request {i+1}: {'ALLOWED' if allowed else 'DENIED'}")
        if not allowed:
            time.sleep(1)  # Wait for token refill
    
    # Test sliding window limiter
    window_limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=5)
    
    print("\nSliding Window Tests:")
    for i in range(6):
        allowed = window_limiter.is_allowed()
        print(f"Request {i+1}: {'ALLOWED' if allowed else 'DENIED'}")
        time.sleep(1)
    
    # Test rate limit manager
    manager = get_rate_limit_manager()
    
    print("\nRate Limit Manager Tests:")
    for i in range(5):
        allowed = manager.is_allowed('api_general')
        print(f"API Request {i+1}: {'ALLOWED' if allowed else 'DENIED'}")
    
    # Test decorator
    @rate_limit('test_decorator', tokens=1)
    def test_function():
        return "Success"
    
    print("\nDecorator Tests:")
    for i in range(3):
        try:
            result = test_function()
            print(f"Function call {i+1}: {result}")
        except RateLimitExceeded as e:
            print(f"Function call {i+1}: Rate limited - {e}")
    
    # Test health monitoring
    health_monitor = get_health_monitor()
    
    for i in range(10):
        allowed = manager.is_allowed('api_general')
        health_monitor.record_usage('api_general', allowed)
    
    health_metrics = health_monitor.get_health_metrics('api_general')
    print(f"\nHealth Metrics: {health_metrics}")
    
    print("Rate limiter tests completed!")

if __name__ == "__main__":
    test_rate_limiters()