"""
Minimal bootstrap for notebooks to import config module.

This tiny helper resolves the chicken-and-egg problem where notebooks
need to import 'config' but can't because project root isn't in sys.path yet.

Usage in notebooks (first cell):
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path.cwd().parent
    sys.path.insert(0, str(project_root))

    # Now import config normally
    from config import setup_notebook
    setup_notebook()

Or use this one-liner alternative:
    exec(open('../config/notebook_setup.py').read())
"""

import sys
from pathlib import Path

# Find project root (go up from config/ directory)
project_root = Path(__file__).parent.parent

# Add to Python path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"✓ Added to Python path: {project_root}")
print(f"✓ You can now import from 'config' and 'modules'")