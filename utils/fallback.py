"""
Fallback handler for gracefully degrading between primary and backup services.
Written to feel natural and human-readable.
"""

import logging
from typing import Any, Callable, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FallbackChain:
    """
    Try multiple strategies in order until one succeeds.
    
    Example:
        chain = FallbackChain("search")
        chain.add_strategy("Tavily", lambda q: tavily.search(q))
        chain.add_strategy("DuckDuckGo", lambda q: ddgs.search(q))
        
        results = chain.execute("Apple Inc news")
    """
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.strategies: List[tuple[str, Callable]] = []
    
    def add_strategy(self, strategy_name: str, func: Callable):
        """Add a fallback strategy"""
        self.strategies.append((strategy_name, func))
        return self  # Allow chaining
    
    def execute(self, *args, **kwargs) -> Optional[T]:
        """Execute strategies in order until one succeeds"""
        if not self.strategies:
            raise ValueError(f"No strategies configured for {self.name}")
        
        last_error = None
        
        for strategy_name, func in self.strategies:
            try:
                logger.info(f"{self.name}: Trying {strategy_name}...")
                result = func(*args, **kwargs)
                
                # Check if result is meaningful (not None or empty)
                if result:
                    logger.info(f"{self.name}: {strategy_name} succeeded")
                    return result
                else:
                    logger.warning(f"{self.name}: {strategy_name} returned empty result")
            
            except Exception as e:
                last_error = e
                logger.warning(f"{self.name}: {strategy_name} failed - {e}")
                continue
        
        # All strategies failed
        logger.error(f"{self.name}: All fallback strategies exhausted")
        if last_error:
            raise last_error
        
        return None


def with_fallback(*funcs: Callable) -> Callable:
    """
    Decorator to try multiple functions in order.
    
    @with_fallback(primary_search, backup_search)
    def search(query):
        pass  # This body is ignored; first successful func is used
    """
    def decorator(original_func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            chain = FallbackChain(original_func.__name__)
            
            for i, func in enumerate(funcs, 1):
                chain.add_strategy(f"Strategy {i}", func)
            
            return chain.execute(*args, **kwargs)
        
        return wrapper
    return decorator


class ServiceFallback:
    """
    Intelligent fallback with health tracking.
    Remembers which services are working and prioritizes them.
    """
    
    def __init__(self, name: str = "service"):
        self.name = name
        self.services: List[dict] = []
        self.failure_counts: dict[str, int] = {}
        self.success_counts: dict[str, int] = {}
    
    def register(self, service_name: str, func: Callable, priority: int = 0):
        """Register a service with priority (lower = try first)"""
        self.services.append({
            "name": service_name,
            "func": func,
            "priority": priority
        })
        self.failure_counts[service_name] = 0
        self.success_counts[service_name] = 0
        
        # Sort by priority
        self.services.sort(key=lambda s: s["priority"])
    
    def call(self, *args, **kwargs) -> Optional[Any]:
        """Call services in priority order, tracking health"""
        
        for service in self.services:
            name = service["name"]
            func = service["func"]
            
            # Skip if service has too many recent failures
            if self.failure_counts[name] > 3:
                logger.debug(f"Skipping {name} due to recent failures")
                continue
            
            try:
                result = func(*args, **kwargs)
                
                if result:
                    self.success_counts[name] += 1
                    self.failure_counts[name] = max(0, self.failure_counts[name] - 1)
                    logger.info(f"{self.name}: {name} succeeded")
                    return result
            
            except Exception as e:
                self.failure_counts[name] += 1
                logger.warning(f"{self.name}: {name} failed - {e}")
                continue
        
        logger.error(f"{self.name}: All services failed")
        return None
    
    def health_report(self) -> dict:
        """Get health statistics for all services"""
        return {
            name: {
                "successes": self.success_counts.get(name, 0),
                "failures": self.failure_counts.get(name, 0),
                "health": self.success_counts.get(name, 0) / 
                         max(1, self.success_counts.get(name, 0) + self.failure_counts.get(name, 0))
            }
            for name in self.success_counts.keys()
        }


def safe_fallback(default_value: T):
    """
    Return default value if function fails.
    
    @safe_fallback([])
    def get_results():
        return api.fetch()  # If this fails, returns []
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result if result is not None else default_value
            except Exception as e:
                logger.warning(f"{func.__name__} failed, using default: {e}")
                return default_value
        
        return wrapper
    return decorator
