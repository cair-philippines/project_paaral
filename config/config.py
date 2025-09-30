"""
Configuration module for Project Paaral

Provides portable path resolution and notebook setup utilities for running
notebooks in any environment without hardcoded paths.

Usage in notebooks:
    from config import setup_notebook, get_path, get_psgc_path

    setup_notebook()
    psgc_path = get_path('psgc_shapefiles')
    adm4_file = get_psgc_path('adm4')
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union


class Config:
    """Configuration manager with automatic project root detection."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.

        Parameters:
        -----------
        config_path : Path, optional
            Path to config.json. If None, auto-detects from this file's location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        self.config_path = Path(config_path)
        self.project_root = self._find_project_root()
        self._load_config()
        self._resolve_paths()

    def _find_project_root(self) -> Path:
        """
        Find project root by looking for marker directories.

        Searches upward from current file location for a directory containing
        'modules', 'data', and 'notebooks' subdirectories.

        Returns:
        --------
        Path to project root
        """
        # Start from config directory's parent
        current = Path(__file__).parent.parent

        # Check if this is already the root (has required markers)
        markers = ['modules', 'data', 'notebooks']
        if all((current / marker).exists() for marker in markers):
            return current

        # Search upward for project root
        for parent in [current] + list(current.parents):
            if all((parent / marker).exists() for marker in markers):
                return parent

        # Fallback: use config directory's parent
        return Path(__file__).parent.parent

    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    def _resolve_paths(self):
        """Convert relative paths to absolute Path objects."""
        self.paths = {}
        for key, relative_path in self.config.get("paths", {}).items():
            self.paths[key] = self.project_root / relative_path

            # Create output directories if they don't exist
            if key in ['output', 'results', 'reconciled', 'processed_data']:
                self.paths[key].mkdir(exist_ok=True, parents=True)

    def get_path(self, path_name: str) -> Path:
        """
        Get absolute path for a configured path name.

        Parameters:
        -----------
        path_name : str
            Name of path from config.json (e.g., 'data', 'psgc_shapefiles')

        Returns:
        --------
        Absolute Path object
        """
        if path_name not in self.paths:
            raise KeyError(f"Path '{path_name}' not found in configuration. "
                         f"Available: {list(self.paths.keys())}")
        return self.paths[path_name]

    def get_psgc_path(self, admin_level: str) -> Path:
        """
        Get path to PSGC shapefile for specified admin level.

        Parameters:
        -----------
        admin_level : str
            Admin level: 'adm0', 'adm1', 'adm2', 'adm3', or 'adm4'

        Returns:
        --------
        Absolute path to shapefile
        """
        if admin_level not in self.config.get("psgc", {}):
            raise KeyError(f"PSGC admin level '{admin_level}' not found. "
                         f"Available: {list(self.config.get('psgc', {}).keys())}")

        shapefile_name = self.config["psgc"][admin_level]
        return self.get_path('psgc_shapefiles') / shapefile_name

    def get_data_path(self, *subpath: str) -> Path:
        """
        Get path within data directory.

        Parameters:
        -----------
        *subpath : str
            Subdirectory/file path components

        Returns:
        --------
        Absolute path within data directory

        Example:
        --------
        >>> config.get_data_path('public', 'enrollment.csv')
        Path('/path/to/project/data/public/enrollment.csv')
        """
        return self.get_path('data').joinpath(*subpath)

    def get_output_path(self, *subpath: str) -> Path:
        """
        Get path within output directory.

        Parameters:
        -----------
        *subpath : str
            Subdirectory/file path components

        Returns:
        --------
        Absolute path within output directory
        """
        path = self.get_path('output').joinpath(*subpath)
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def get_setting(self, setting_name: str, default: Any = None) -> Any:
        """Get experiment setting with optional default."""
        return self.config.get("experiment_settings", {}).get(setting_name, default)

    def get_file_pattern(self, pattern_name: str) -> str:
        """Get file pattern from configuration."""
        return self.config.get("file_patterns", {}).get(pattern_name, "*")

    def list_files(self, path_name: str, pattern: str = "*") -> list:
        """
        List files in a configured directory.

        Parameters:
        -----------
        path_name : str
            Name of configured path
        pattern : str
            Glob pattern (default: "*")

        Returns:
        --------
        List of Path objects
        """
        directory = self.get_path(path_name)
        return sorted(directory.glob(pattern))

    def setup_logging(self) -> logging.Logger:
        """
        Configure logging based on config settings.

        Returns:
        --------
        Configured logger
        """
        log_config = self.config.get("logging", {})

        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get("format", "%(levelname)s - %(message)s"),
            datefmt=log_config.get("date_format", "%Y-%m-%d %H:%M:%S")
        )

        return logging.getLogger(__name__)

    def __repr__(self):
        return f"Config(project_root={self.project_root})"


# Global config instance
_config_instance = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def setup_notebook(verbose: bool = True) -> Dict[str, Path]:
    """
    Setup notebook environment for portable execution.

    This function:
    1. Changes working directory to project root
    2. Adds project root to sys.path for module imports
    3. Returns commonly used paths

    Parameters:
    -----------
    verbose : bool
        Print setup information (default: True)

    Returns:
    --------
    Dictionary with commonly used paths:
        - project_root
        - data
        - modules
        - notebooks
        - output
        - psgc_shapefiles

    Example:
    --------
    >>> from config import setup_notebook
    >>> paths = setup_notebook()
    >>> print(paths['psgc_shapefiles'])
    """
    config = get_config()

    # Change to project root
    os.chdir(config.project_root)

    # Add project root to Python path if not already there
    project_root_str = str(config.project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    if verbose:
        print(f"✓ Project root: {config.project_root}")
        print(f"✓ Working directory: {os.getcwd()}")
        print(f"✓ Python path updated")

    # Return commonly used paths
    return {
        'project_root': config.project_root,
        'data': config.get_path('data'),
        'modules': config.get_path('modules'),
        'notebooks': config.get_path('notebooks'),
        'output': config.get_path('output'),
        'psgc_shapefiles': config.get_path('psgc_shapefiles'),
    }


def get_path(path_name: str) -> Path:
    """Convenience function to get configured path."""
    return get_config().get_path(path_name)


def get_psgc_path(admin_level: str) -> Path:
    """Convenience function to get PSGC shapefile path."""
    return get_config().get_psgc_path(admin_level)


def get_data_path(*subpath: str) -> Path:
    """Convenience function to get path within data directory."""
    return get_config().get_data_path(*subpath)


def get_output_path(*subpath: str) -> Path:
    """Convenience function to get path within output directory."""
    return get_config().get_output_path(*subpath)


# Expose PROJECT_ROOT constant for backward compatibility
PROJECT_ROOT = get_config().project_root