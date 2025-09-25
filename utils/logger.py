"""
Logging Utilities
=================

Centralized logging configuration and utilities for the rail traffic AI project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    log_dir: str = "logs"
) -> None:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path (if None, uses timestamp-based filename)
        log_format: Custom log format string
        log_dir: Directory to store log files
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Default log format
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # Default log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"rail_traffic_ai_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    _configure_specific_loggers()
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def _configure_specific_loggers():
    """Configure specific loggers with appropriate levels."""
    # Reduce verbosity for external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Set our package loggers to INFO by default
    logging.getLogger('rail_traffic_ai').setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLogger:
    """
    Structured logger for JSON-formatted logs.
    
    Useful for production environments where logs need to be parsed
    by log aggregation systems.
    """
    
    def __init__(self, name: str):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Add JSON formatter if not already present
        if not any(isinstance(h, logging.StreamHandler) and 
                  isinstance(h.formatter, JSONFormatter) for h in self.logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method."""
        extra = {
            'timestamp': datetime.now().isoformat(),
            'level': logging.getLevelName(level),
            'message': message,
            **kwargs
        }
        self.logger.log(level, message, extra=extra)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """
    Logger for performance metrics and timing.
    """
    
    def __init__(self, name: str):
        """
        Initialize performance logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(f"{name}.performance")
        self.timers = {}
    
    def start_timer(self, operation: str):
        """
        Start timing an operation.
        
        Args:
            operation: Operation name
        """
        self.timers[operation] = datetime.now()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str, **kwargs):
        """
        End timing an operation and log the duration.
        
        Args:
            operation: Operation name
            **kwargs: Additional data to log
        """
        if operation not in self.timers:
            self.logger.warning(f"Timer not found for operation: {operation}")
            return
        
        start_time = self.timers[operation]
        duration = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                'operation': operation,
                'duration_seconds': duration,
                **kwargs
            }
        )
        
        del self.timers[operation]
        return duration
    
    def log_metric(self, metric_name: str, value: float, **kwargs):
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            **kwargs: Additional data to log
        """
        self.logger.info(
            f"Performance metric: {metric_name}",
            extra={
                'metric_name': metric_name,
                'metric_value': value,
                **kwargs
            }
        )


class DataLogger:
    """
    Logger for data pipeline operations.
    """
    
    def __init__(self, name: str):
        """
        Initialize data logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(f"{name}.data")
    
    def log_data_ingestion(self, source: str, record_count: int, **kwargs):
        """
        Log data ingestion event.
        
        Args:
            source: Data source name
            record_count: Number of records ingested
            **kwargs: Additional data to log
        """
        self.logger.info(
            f"Data ingested from {source}",
            extra={
                'operation': 'data_ingestion',
                'source': source,
                'record_count': record_count,
                **kwargs
            }
        )
    
    def log_data_processing(self, operation: str, input_count: int, output_count: int, **kwargs):
        """
        Log data processing event.
        
        Args:
            operation: Processing operation name
            input_count: Number of input records
            output_count: Number of output records
            **kwargs: Additional data to log
        """
        self.logger.info(
            f"Data processed: {operation}",
            extra={
                'operation': 'data_processing',
                'processing_operation': operation,
                'input_count': input_count,
                'output_count': output_count,
                'success_rate': output_count / input_count if input_count > 0 else 0,
                **kwargs
            }
        )
    
    def log_data_quality(self, dataset_name: str, quality_metrics: Dict[str, Any], **kwargs):
        """
        Log data quality metrics.
        
        Args:
            dataset_name: Name of the dataset
            quality_metrics: Dictionary of quality metrics
            **kwargs: Additional data to log
        """
        self.logger.info(
            f"Data quality check: {dataset_name}",
            extra={
                'operation': 'data_quality',
                'dataset_name': dataset_name,
                'quality_metrics': quality_metrics,
                **kwargs
            }
        )
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float], **kwargs):
        """
        Log model performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            **kwargs: Additional data to log
        """
        self.logger.info(
            f"Model performance: {model_name}",
            extra={
                'operation': 'model_performance',
                'model_name': model_name,
                'performance_metrics': metrics,
                **kwargs
            }
        )


# Convenience functions for common logging patterns
def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper


def log_execution_time(operation_name: str):
    """Decorator to log execution time of functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = datetime.now()
            logger.debug(f"Starting {operation_name}")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"{operation_name} completed in {duration:.2f} seconds")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"{operation_name} failed after {duration:.2f} seconds: {e}")
                raise
        return wrapper
    return decorator
