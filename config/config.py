import json
from pathlib import Path

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        
        self.config_path = Path(config_path)
        self.base_dir = Path(__file__).parent.parent  # Go up one level from config/
        self._load_config()
        self._resolve_paths()
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def _resolve_paths(self):
        """Convert relative paths to absolute Path objects"""
        self.paths = {}
        for key, relative_path in self.config["data_paths"].items():
            self.paths[key] = self.base_dir / relative_path
            # Create directory if it doesn't exist and it's an output directory
            if key in ['results']:
                self.paths[key].mkdir(exist_ok=True, parents=True)
    
    def get_path(self, path_name):
        """Get a specific path by name"""
        if path_name not in self.paths:
            raise KeyError(f"Path '{path_name}' not found in configuration")
        return self.paths[path_name]
    
    def get_setting(self, setting_name):
        """Get experiment setting"""
        return self.config["experiment_settings"].get(setting_name)
    
    def get_file_pattern(self, pattern_name):
        """Get file pattern"""
        return self.config["file_patterns"].get(pattern_name)
    
    def list_input_files(self, path_name, pattern=None):
        """List files in a directory with optional pattern"""
        directory = self.get_path(path_name)
        if pattern is None:
            pattern = "*"
        return list(directory.glob(pattern))