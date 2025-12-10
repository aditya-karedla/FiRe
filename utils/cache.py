"""
Simple file-based caching with TTL support.
Keeps things fast and avoids hammering APIs unnecessarily.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

from config.settings import settings

logger = logging.getLogger(__name__)


class FileCache:
    """
    Simple file-based cache with expiration.
    
    Usage:
        cache = FileCache("my_data", ttl=3600)
        
        # Set
        cache.set("key", {"some": "data"})
        
        # Get
        data = cache.get("key")  # Returns None if expired or missing
    """
    
    def __init__(self, namespace: str = "default", ttl: int = 3600):
        """
        Args:
            namespace: Logical grouping for cache files
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.namespace = namespace
        self.ttl = ttl
        self.cache_dir = settings.CACHE_DIR / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        """Get file path for a cache key"""
        # Simple sanitization
        safe_key = "".join(c if c.isalnum() or c in "_-" else "_" for c in key)
        return self.cache_dir / f"{safe_key}.cache"
    
    def _get_meta_path(self, key: str) -> Path:
        """Get metadata file path"""
        return self._get_path(key).with_suffix(".meta")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store value in cache"""
        cache_path = self._get_path(key)
        meta_path = self._get_meta_path(key)
        
        try:
            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Save metadata
            metadata = {
                "timestamp": time.time(),
                "ttl": ttl or self.ttl,
                "key": key
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.debug(f"Cached {key} in {self.namespace}")
        
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache if not expired"""
        cache_path = self._get_path(key)
        meta_path = self._get_meta_path(key)
        
        if not cache_path.exists() or not meta_path.exists():
            return None
        
        try:
            # Check if expired
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            age = time.time() - metadata["timestamp"]
            if age > metadata["ttl"]:
                logger.debug(f"Cache expired for {key} (age: {age:.0f}s)")
                self.delete(key)
                return None
            
            # Load data
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            
            logger.debug(f"Cache hit for {key} (age: {age:.0f}s)")
            return value
        
        except Exception as e:
            logger.warning(f"Failed to read cache for {key}: {e}")
            return None
    
    def delete(self, key: str):
        """Remove cached value"""
        cache_path = self._get_path(key)
        meta_path = self._get_meta_path(key)
        
        cache_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    
    def clear(self):
        """Clear all cached values in this namespace"""
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file():
                file_path.unlink()
        
        logger.info(f"Cleared cache for {self.namespace}")
    
    def get_or_compute(self, key: str, compute_func: Callable, ttl: Optional[int] = None) -> Any:
        """Get cached value or compute and cache it"""
        value = self.get(key)
        
        if value is not None:
            return value
        
        logger.debug(f"Cache miss for {key}, computing...")
        value = compute_func()
        self.set(key, value, ttl)
        
        return value


def cached(namespace: str = "default", ttl: int = 3600, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    @cached("api_calls", ttl=1800)
    def fetch_data(param):
        return expensive_api_call(param)
    """
    cache = FileCache(namespace, ttl)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Simple key from function name and args
                cache_key = f"{func.__name__}_{hash((args, frozenset(kwargs.items())))}"
            
            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        # Add cache control methods
        wrapper.clear_cache = cache.clear
        wrapper.cache = cache
        
        return wrapper
    
    return decorator


class JsonFileCache:
    """
    Simple JSON file cache for human-readable cached data.
    Useful for debugging and inspection.
    """
    
    def __init__(self, file_path: Union[str, Path], ttl: int = 86400):
        self.file_path = Path(file_path)
        self.ttl = ttl
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, data: Any):
        """Save data with timestamp"""
        cache_data = {
            "timestamp": time.time(),
            "data": data
        }
        
        with open(self.file_path, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        
        logger.debug(f"Saved to {self.file_path}")
    
    def load(self) -> Optional[Any]:
        """Load data if not expired"""
        if not self.file_path.exists():
            return None
        
        try:
            with open(self.file_path, 'r') as f:
                cache_data = json.load(f)
            
            age = time.time() - cache_data["timestamp"]
            
            if age > self.ttl:
                logger.debug(f"Cache expired: {self.file_path} (age: {age:.0f}s)")
                return None
            
            logger.debug(f"Loaded from {self.file_path} (age: {age:.0f}s)")
            return cache_data["data"]
        
        except Exception as e:
            logger.warning(f"Failed to load cache from {self.file_path}: {e}")
            return None
    
    def is_fresh(self) -> bool:
        """Check if cache exists and is not expired"""
        return self.load() is not None
