"""
Configuration Management
========================

Centralized configuration management for the rail traffic AI project.
Supports YAML, JSON, and environment variable configurations.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    name: str = "rail_traffic_ai"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class APIConfig:
    """API configuration settings."""
    base_url: str = "https://api.railway.gov.in"
    api_key: str = ""
    rate_limit_delay: float = 1.0
    max_retries: int = 3
    timeout: int = 30


@dataclass
class ModelConfig:
    """ML model configuration settings."""
    model_dir: str = "models"
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10


@dataclass
class DataConfig:
    """Data pipeline configuration settings."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    external_dir: str = "data/external"
    synthetic_dir: str = "data/synthetic"
    cache_ttl: int = 3600
    max_file_size_mb: int = 100


class Config:
    """
    Centralized configuration management class.
    
    Supports loading from YAML files, JSON files, and environment variables.
    Provides type-safe access to configuration values with defaults.
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
        """
        self._config = {}
        self._config_file = config_file
        
        # Load default configuration
        self._load_defaults()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Load from environment variables
        self._load_from_env()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self._config = {
            # Database settings
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'rail_traffic_ai',
                'user': 'postgres',
                'password': '',
                'pool_size': 10,
                'max_overflow': 20
            },
            
            # API settings
            'ntes': {
                'base_url': 'https://api.railway.gov.in',
                'api_key': '',
                'rate_limit_delay': 1.0,
                'max_retries': 3,
                'timeout': 30,
                'cache_ttl': 300
            },
            
            # Weather API settings
            'weather': {
                'api_key': '',
                'base_url': 'https://api.openweathermap.org/data/2.5',
                'cache_ttl': 3600
            },
            
            # Holiday API settings
            'holidays': {
                'api_key': '',
                'base_url': 'https://date.nager.at/api/v3',
                'cache_ttl': 86400
            },
            
            # Data pipeline settings
            'data': {
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'external_dir': 'data/external',
                'synthetic_dir': 'data/synthetic',
                'cache_ttl': 3600,
                'max_file_size_mb': 100
            },
            
            # ML model settings
            'models': {
                'model_dir': 'models',
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'validation_split': 0.2,
                'early_stopping_patience': 10
            },
            
            # Optimization settings
            'optimization': {
                'solver_timeout': 300,
                'max_iterations': 1000,
                'tolerance': 1e-6,
                'parallel_workers': 4
            },
            
            # Dashboard settings
            'dashboard': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False,
                'secret_key': 'dev-secret-key'
            },
            
            # Logging settings
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs',
                'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'max_file_size_mb': 10,
                'backup_count': 5
            },
            
            # External data settings
            'external': {
                'cache_ttl': 3600,
                'update_interval': 1800,
                'max_retries': 3
            }
        }
    
    def load_from_file(self, config_file: Union[str, Path]):
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                    return
            
            # Merge with existing configuration
            self._merge_config(file_config)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """
        Merge new configuration with existing configuration.
        
        Args:
            new_config: New configuration dictionary
        """
        def merge_dict(base: Dict, update: Dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self._config, new_config)
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'DATABASE_HOST': 'database.host',
            'DATABASE_PORT': 'database.port',
            'DATABASE_NAME': 'database.name',
            'DATABASE_USER': 'database.user',
            'DATABASE_PASSWORD': 'database.password',
            'NTES_API_KEY': 'ntes.api_key',
            'WEATHER_API_KEY': 'weather.api_key',
            'HOLIDAY_API_KEY': 'holidays.api_key',
            'LOG_LEVEL': 'logging.level',
            'DASHBOARD_PORT': 'dashboard.port',
            'DASHBOARD_DEBUG': 'dashboard.debug'
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self.set(config_path, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'database.host')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'database')
            
        Returns:
            Configuration section dictionary
        """
        return self.get(section, {})
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration as dataclass."""
        db_config = self.get_section('database')
        return DatabaseConfig(**db_config)
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration as dataclass."""
        api_config = self.get_section('ntes')
        return APIConfig(**api_config)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration as dataclass."""
        model_config = self.get_section('models')
        return ModelConfig(**model_config)
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration as dataclass."""
        data_config = self.get_section('data')
        return DataConfig(**data_config)
    
    def save_to_file(self, config_file: Union[str, Path]):
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to save configuration file
        """
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(self._config, f, indent=2)
                else:
                    logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                    return
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
    
    def validate(self) -> List[str]:
        """
        Validate configuration values.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate database configuration
        db_config = self.get_section('database')
        if not db_config.get('host'):
            errors.append("Database host is required")
        if not isinstance(db_config.get('port'), int) or db_config.get('port') <= 0:
            errors.append("Database port must be a positive integer")
        
        # Validate API configuration
        ntes_config = self.get_section('ntes')
        if not ntes_config.get('base_url'):
            errors.append("NTES base URL is required")
        if not isinstance(ntes_config.get('rate_limit_delay'), (int, float)) or ntes_config.get('rate_limit_delay') < 0:
            errors.append("Rate limit delay must be a non-negative number")
        
        # Validate data configuration
        data_config = self.get_section('data')
        if not isinstance(data_config.get('cache_ttl'), int) or data_config.get('cache_ttl') <= 0:
            errors.append("Cache TTL must be a positive integer")
        
        # Validate model configuration
        model_config = self.get_section('models')
        if not isinstance(model_config.get('learning_rate'), (int, float)) or model_config.get('learning_rate') <= 0:
            errors.append("Learning rate must be a positive number")
        if not isinstance(model_config.get('epochs'), int) or model_config.get('epochs') <= 0:
            errors.append("Epochs must be a positive integer")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style assignment."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return self.get(key) is not None


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance.
    
    Returns:
        Global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """
    Set global configuration instance.
    
    Args:
        config: Configuration instance to set as global
    """
    global _global_config
    _global_config = config


def load_config(config_file: Union[str, Path]) -> Config:
    """
    Load configuration from file and set as global.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Loaded configuration instance
    """
    config = Config(config_file)
    set_config(config)
    return config
