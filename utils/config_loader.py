"""
Utility for loading YAML configuration files.
"""

import yaml
from pathlib import Path


def load_config(config_path: str | Path | None = None) -> dict:
    """Load config.yaml from the given path or default location (project root)."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
