"""Plain-YAML config loading.

Loads `conf/config.yaml` (with `extends:` support for future use) and applies
Hydra-style dotted `key=value` CLI overrides, e.g. `replay.speed=0.5`. Replaces
Hydra + OmegaConf: `ConfigDict` is a plain dict with recursive dotted attribute
access, so existing `cfg.robot.ip` / `cfg.get("gripper", {})` call sites are
unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


class ConfigDict(dict):
    """dict subclass with recursive dotted attribute access."""

    def __init__(self, data=None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = _wrap(v)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = _wrap(value)


def _wrap(value):
    if isinstance(value, dict) and not isinstance(value, ConfigDict):
        return ConfigDict(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def nested_override_from_keypath(key_path: str, value: Any) -> dict[str, Any]:
    """Build a nested dict override from a dotted key path."""
    parts = [part.strip() for part in key_path.split(".") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid override key path: {key_path!r}")

    override: dict[str, Any] | Any = value
    for part in reversed(parts):
        override = {part: override}
    return override


def parse_override_entry(override_entry: str) -> dict[str, Any]:
    """Parse a KEY=VALUE override string into a nested config dict."""
    if "=" not in override_entry:
        raise ValueError(
            f"Invalid override {override_entry!r}. Expected KEY=VALUE, "
            "for example replay.speed=0.5"
        )
    key_path, raw_value = override_entry.split("=", 1)
    parsed_value = yaml.safe_load(raw_value)
    return nested_override_from_keypath(key_path, parsed_value)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply a sequence of KEY=VALUE dotted-path overrides to a config dict."""
    merged = dict(config)
    for override_entry in overrides:
        merged = deep_merge_dicts(merged, parse_override_entry(override_entry))
    return merged


def _load_yaml_with_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    cfg_path = path.resolve()
    if seen is None:
        seen = set()
    if cfg_path in seen:
        chain = " -> ".join(str(p) for p in [*seen, cfg_path])
        raise ValueError(f"Cyclic config extends detected: {chain}")

    seen.add(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Top-level config must be a mapping: {cfg_path}")

    extends = raw.pop("extends", None)
    if extends is None:
        seen.remove(cfg_path)
        return raw

    if isinstance(extends, (str, Path)):
        extends_list = [extends]
    elif isinstance(extends, list):
        extends_list = extends
    else:
        raise TypeError(
            f"'extends' must be a string/path or list in {cfg_path}, got {type(extends)}"
        )

    merged: dict[str, Any] = {}
    for entry in extends_list:
        if not isinstance(entry, (str, Path)):
            raise TypeError(
                f"Each extends entry must be a string/path in {cfg_path}, got {type(entry)}"
            )
        parent_candidate = Path(entry)
        if not parent_candidate.is_absolute():
            parent_candidate = (cfg_path.parent / parent_candidate).resolve()
        parent_cfg = _load_yaml_with_extends(parent_candidate, seen=seen)
        merged = deep_merge_dicts(merged, parent_cfg)

    merged = deep_merge_dicts(merged, raw)
    seen.remove(cfg_path)
    return merged


def load_app_config(script_file: str | Path, argv: list[str] | None = None) -> ConfigDict:
    """Load <script_dir>/conf/config.yaml and apply dotted key=value CLI overrides."""
    if argv is None:
        argv = sys.argv[1:]
    config_path = Path(script_file).resolve().parent / "conf" / "config.yaml"
    base = _load_yaml_with_extends(config_path)
    merged = apply_overrides(base, argv)
    return ConfigDict(merged)
