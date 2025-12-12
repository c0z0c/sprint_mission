# -*- coding: utf-8 -*-
"""MNIST History Management Package

This package provides functionality for storing, querying, and managing prediction records.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.history.FileHistoryManager import FileHistoryManager
from src.history.history_manager import generate_dummy_history_data
from src.history.HistoryManager import HistoryManager
from src.history.HistoryRecord import HistoryRecord

__all__ = [
    "HistoryRecord",
    "HistoryManager",
    "FileHistoryManager",
    "generate_dummy_history_data",
]
