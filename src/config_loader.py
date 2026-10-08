import yaml
from pathlib import Path
import os

CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'

class ConfigLoader:
    _instance = None
    _config = None

    @classmethod
    def get_config(cls):
        if cls._config is None:
            if not os.path.exists(CONFIG_PATH):
                raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
            with open(CONFIG_PATH, 'r') as f:
                cls._config = yaml.safe_load(f)
            cls.validate_config(cls._config)
        return cls._config

    @staticmethod
    def validate_config(config):
        required_keys = [
            'config_version', 'demo_area', 'canonical_times',
            'building_height_per_floor', 'fallback_building_floors',
            'risk_normalization_method', 'risk_class_ranges',
            'solar', 'shadow', 'temperature_proxy', 'exposure', 'fallback', 'computation_mode'
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

def get_config():
    return ConfigLoader.get_config()
