# -*- coding: utf-8 -*-
"""Utility for loading scripts by file path.

The pipeline now calls the modules package directly and no longer relies on
this utility. It is kept for loading external experimental modules or plugin
scripts in the future.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module(script_path: Path, module_name: str):
    """Load a Python module from the given path."""
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
