"""Tests for configuration management."""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import (
    get_config,
    get_nested_config,
    load_config,
    merge_configs,
    save_config,
)


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_config_valid(self):
        """Test loading a valid config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            config_data = {"key1": "value1", "key2": 123, "nested": {"a": 1, "b": 2}}

            with open(config_path, "w") as f:
                yaml.safe_dump(config_data, f)

            loaded = load_config(config_path)

            assert loaded == config_data

    def test_load_config_file_not_found(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/path/config.yaml"))

    def test_load_config_string_path(self):
        """Test loading config with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.yaml"
            config_data = {"test": "value"}

            with open(config_path, "w") as f:
                yaml.safe_dump(config_data, f)

            # Pass string instead of Path
            loaded = load_config(config_path)
            assert loaded["test"] == "value"


class TestGetConfig:
    """Tests for config retrieval with caching."""

    def test_get_config(self):
        """Test getting config by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_data = {"setting": "value"}

            with open(config_dir / "myconfig.yaml", "w") as f:
                yaml.safe_dump(config_data, f)

            loaded = get_config("myconfig", config_dir=config_dir)
            assert loaded["setting"] == "value"

    def test_get_config_caching(self):
        """Test that config is cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_data = {"value": 1}

            with open(config_dir / "cached.yaml", "w") as f:
                yaml.safe_dump(config_data, f)

            # First load
            get_config("cached", config_dir=config_dir)

            # Modify the file
            with open(config_dir / "cached.yaml", "w") as f:
                yaml.safe_dump({"value": 2}, f)

            # Should return cached version
            config2 = get_config("cached", config_dir=config_dir)
            assert config2["value"] == 1

            # With reload=True, should get new value
            config3 = get_config("cached", config_dir=config_dir, reload=True)
            assert config3["value"] == 2


class TestGetNestedConfig:
    """Tests for nested config access."""

    def test_get_nested_config_simple(self):
        """Test getting a simple nested value."""
        config = {"level1": {"level2": {"level3": "value"}}}
        result = get_nested_config(config, "level1.level2.level3")
        assert result == "value"

    def test_get_nested_config_top_level(self):
        """Test getting a top-level value."""
        config = {"key": "value"}
        result = get_nested_config(config, "key")
        assert result == "value"

    def test_get_nested_config_missing_key(self):
        """Test default value for missing key."""
        config = {"key": "value"}
        result = get_nested_config(config, "missing.key", default="default")
        assert result == "default"

    def test_get_nested_config_default_none(self):
        """Test default None for missing key."""
        config = {"key": "value"}
        result = get_nested_config(config, "missing")
        assert result is None

    def test_get_nested_config_partial_path(self):
        """Test partial path returns intermediate dict."""
        config = {"level1": {"level2": {"a": 1, "b": 2}}}
        result = get_nested_config(config, "level1.level2")
        assert result == {"a": 1, "b": 2}


class TestSaveConfig:
    """Tests for config saving."""

    def test_save_config(self):
        """Test saving config to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "saved.yaml"
            config_data = {"key": "value", "number": 42}

            save_config(config_data, config_path)

            assert config_path.exists()

            with open(config_path) as f:
                loaded = yaml.safe_load(f)

            assert loaded == config_data

    def test_save_config_creates_parent_dirs(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nested" / "dirs" / "config.yaml"
            config_data = {"test": True}

            save_config(config_data, config_path)

            assert config_path.exists()


class TestMergeConfigs:
    """Tests for config merging."""

    def test_merge_configs_simple(self):
        """Test simple config merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = merge_configs(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_configs_nested(self):
        """Test nested config merge."""
        base = {"outer": {"inner1": 1, "inner2": 2}}
        override = {"outer": {"inner2": 20, "inner3": 30}}

        result = merge_configs(base, override)

        assert result == {"outer": {"inner1": 1, "inner2": 20, "inner3": 30}}

    def test_merge_configs_deep_nested(self):
        """Test deeply nested config merge."""
        base = {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}
        override = {"l1": {"l2": {"l3": {"b": 20}}}}

        result = merge_configs(base, override)

        assert result["l1"]["l2"]["l3"] == {"a": 1, "b": 20}

    def test_merge_configs_base_unchanged(self):
        """Test that base config is not modified."""
        base = {"a": 1}
        override = {"a": 2}

        merge_configs(base, override)

        assert base == {"a": 1}

    def test_merge_configs_override_replaces_non_dict(self):
        """Test that non-dict values are replaced, not merged."""
        base = {"key": [1, 2, 3]}
        override = {"key": [4, 5]}

        result = merge_configs(base, override)

        assert result["key"] == [4, 5]
