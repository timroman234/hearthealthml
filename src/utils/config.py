"""Configuration management."""

from pathlib import Path
from typing import Any, cast

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global config cache
_config_cache: dict[str, dict] = {}


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = cast(dict[str, Any], yaml.safe_load(f))

    logger.info(f"Loaded config from {config_path}")
    return config


def get_config(
    config_name: str = "config",
    config_dir: Path | None = None,
    reload: bool = False,
) -> dict:
    """Get configuration by name with caching.

    Args:
        config_name: Name of config (without .yaml extension).
        config_dir: Directory containing config files. Defaults to 'configs/'.
        reload: Force reload from disk.

    Returns:
        Configuration dictionary.
    """
    if config_dir is None:
        config_dir = Path("configs")

    cache_key = f"{config_dir}/{config_name}"

    if not reload and cache_key in _config_cache:
        return _config_cache[cache_key]

    config_path = config_dir / f"{config_name}.yaml"
    config = load_config(config_path)

    _config_cache[cache_key] = config
    return config


def get_nested_config(config: dict, key_path: str, default: Any = None) -> Any:
    """Get nested configuration value using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path (e.g., 'data.raw_path').
        default: Default value if key doesn't exist.

    Returns:
        Configuration value or default.
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def save_config(config: dict, config_path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save config file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {config_path}")


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration.
        override: Override configuration (takes precedence).

    Returns:
        Merged configuration.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
