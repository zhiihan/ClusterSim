# myproject/tests/conftest.py
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Force pytest to act as if run from project root
os.chdir(ROOT)

# Ensure imports work (src layout)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
