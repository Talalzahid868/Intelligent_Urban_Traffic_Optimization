"""
Logging utilities for the integration layer.

Provides consistent logging across all modules with configurable
output formats and destinations.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Module-level logger cache
_loggers = {}


def setup_logger(
    name: str = "traffic_optimization",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to log file.
        console: Whether to log to console.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if name in _loggers:
        return _loggers[name]
    
    logger.setLevel(level)
    
    # Create formatters
    detailed_format = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_format = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_format)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_format)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "traffic_optimization") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class PipelineLogger:
    """
    Structured logger for pipeline operations.
    
    Provides methods for logging pipeline events with consistent
    formatting and optional timing information.
    """
    
    def __init__(self, name: str = "pipeline"):
        self.logger = get_logger(f"traffic_optimization.{name}")
        self._start_time = None
    
    def start(self, operation: str):
        """Log start of an operation."""
        self._start_time = datetime.now()
        self.logger.info(f"Starting: {operation}")
    
    def end(self, operation: str, success: bool = True):
        """Log end of an operation with timing."""
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            status = "completed" if success else "failed"
            self.logger.info(f"{operation} {status} in {elapsed:.3f}s")
        else:
            self.logger.info(f"{operation} {'completed' if success else 'failed'}")
    
    def step(self, message: str):
        """Log a pipeline step."""
        self.logger.info(f"  -> {message}")
    
    def warning(self, message: str):
        """Log a warning."""
        self.logger.warning(message)
    
    def error(self, message: str, exc: Exception = None):
        """Log an error."""
        if exc:
            self.logger.error(f"{message}: {exc}")
        else:
            self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug information."""
        self.logger.debug(message)
    
    def result(self, name: str, value):
        """Log a result value."""
        self.logger.info(f"  {name}: {value}")


def log_module_status(module_name: str, status: str, details: str = None):
    """
    Log module loading/initialization status.
    
    Args:
        module_name: Name of the module.
        status: Status message (e.g., "loaded", "initialized", "error").
        details: Optional additional details.
    """
    logger = get_logger("traffic_optimization.modules")
    
    msg = f"[{module_name}] {status}"
    if details:
        msg += f" - {details}"
    
    if "error" in status.lower():
        logger.error(msg)
    elif "warning" in status.lower():
        logger.warning(msg)
    else:
        logger.info(msg)
