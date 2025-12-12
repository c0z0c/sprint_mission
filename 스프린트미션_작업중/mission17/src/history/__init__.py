# -*- coding: utf-8 -*-
"""MNIST History Management Package

This package provides functionality for storing, querying, and managing prediction records.
"""

import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.history.HistoryRecord import HistoryRecord
from src.history.HistoryManager import HistoryManager
from src.history.FileHistoryManager import FileHistoryManager
from src.history.history_manager import generate_dummy_history_data

__all__ = [
    "HistoryRecord",
    "HistoryManager",
    "FileHistoryManager",
    "generate_dummy_history_data",
]
