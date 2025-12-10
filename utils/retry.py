"""
Retry decorator with exponential backoff for resilient API calls.
Human-like error handling with sensible defaults.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """
    Decorator that retries a function with exponential backoff.
    
    Works for both sync and async functions. Feels natural to use:
    
    @retry(max_attempts=3)
    def fetch_data():
        ...
    
    Args:
        max_attempts: How many times to try before giving up
        backoff_factor: Multiplier for wait time (2.0 = double each time)
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback(attempt_num, error) called before retry
    """
    def decorator(func: Callable) -> Callable:
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Final attempt failed
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    # Calculate wait time with exponential backoff
                    wait_time = backoff_factor ** attempt
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    
                    await asyncio.sleep(wait_time)
            
            # Should never reach here, but just in case
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    wait_time = backoff_factor ** attempt
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    
                    time.sleep(wait_time)
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def rate_limit(calls_per_second: float):
    """
    Simple rate limiter decorator.
    
    @rate_limit(10)  # Max 10 calls per second
    def api_call():
        ...
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # Use list to allow modification in nested function
    
    def decorator(func: Callable) -> Callable:
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RetryStrategy:
    """
    Configurable retry strategy for more complex scenarios.
    
    Usage:
        strategy = RetryStrategy(max_attempts=5, should_retry=lambda e: "timeout" in str(e))
        
        @strategy.retry
        def fetch():
            ...
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        max_wait: float = 60.0,
        should_retry: Optional[Callable[[Exception], bool]] = None,
    ):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.max_wait = max_wait
        self.should_retry = should_retry or (lambda e: True)
    
    def retry(self, func: Callable) -> Callable:
        """Apply retry logic to function"""
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(self.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if not self.should_retry(e) or attempt == self.max_attempts - 1:
                        raise
                    
                    wait = min(self.backoff_factor ** attempt, self.max_wait)
                    logger.info(f"Retry {attempt + 1}/{self.max_attempts} after {wait:.1f}s")
                    await asyncio.sleep(wait)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not self.should_retry(e) or attempt == self.max_attempts - 1:
                        raise
                    
                    wait = min(self.backoff_factor ** attempt, self.max_wait)
                    logger.info(f"Retry {attempt + 1}/{self.max_attempts} after {wait:.1f}s")
                    time.sleep(wait)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
