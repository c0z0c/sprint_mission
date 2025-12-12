# -*- coding: utf-8 -*-
"""MNIST ONNX Modeling Package

This package provides ONNX model management, image preprocessing, and inference functionality for MNIST digit prediction.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.model.DataClass import ModelConfig, PredictionResult
from src.model.DummyDataGenerator import DummyDataGenerator
from src.model.ImagePreprocessor import ImagePreprocessor
from src.model.MNISTPipeline import MNISTPipeline
from src.model.ModelDownloader import ModelDownloader
from src.model.ONNXPredictor import ONNXPredictor

__all__ = [
    "PredictionResult",
    "ModelConfig",
    "DummyDataGenerator",
    "ImagePreprocessor",
    "ModelDownloader",
    "ONNXPredictor",
    "MNISTPipeline",
]
