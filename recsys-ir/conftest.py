"""Root pytest conftest — ensures the project root is on sys.path.

This allows 'from src.X import Y' to work in tests regardless of whether
the package has been installed in editable mode (pip install -e .).
"""

import sys
from pathlib import Path

# Add the project root (directory containing this file) to sys.path
# so that 'import src.common.schema' etc. resolve correctly.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
