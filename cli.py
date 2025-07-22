#!/usr/bin/env python3
"""
EPYC-testing CLI entry point.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.cli.main import main

if __name__ == "__main__":
    main() 