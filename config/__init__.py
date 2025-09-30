"""
Configuration package for Project Paaral

Provides portable configuration and path management for notebooks and scripts.

Quick Start:
    from config import setup_notebook, get_path

    # In notebooks
    setup_notebook()

    # Get paths
    data_path = get_path('data')
    psgc_path = get_path('psgc_shapefiles')
"""

from .config import (
    Config,
    setup_notebook,
    get_config,
    get_path,
    get_psgc_path,
    get_data_path,
    get_output_path,
    PROJECT_ROOT,
)

__all__ = [
    'Config',
    'setup_notebook',
    'get_config',
    'get_path',
    'get_psgc_path',
    'get_data_path',
    'get_output_path',
    'PROJECT_ROOT',
]

__version__ = '1.0.0'