import pytest
from src.config_loader import ConfigLoader

def test_config_loader():
    config = ConfigLoader.get_config()
    assert 'config_version' in config
    assert 'demo_area' in config
    assert 'building_height_per_floor' in config
    assert 'fallback_building_floors' in config

def test_config_validation():
    with pytest.raises(ValueError, match="Missing required config key"):
        ConfigLoader.validate_config({'config_version': '1.0'})
