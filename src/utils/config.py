"""Configuration loader"""

import yaml
from pathlib import Path


class Config:
    """Load and manage configuration"""
    
    @staticmethod
    def load(config_path: str = "config/default_config.yaml") -> dict:
        """Load configuration from YAML file"""
        raise NotImplementedError
