# -*- coding: utf-8 -*-
"""MNIST ONNX Modeling Package

This package provides ONNX model management, image preprocessing, and inference functionality for MNIST digit prediction.
"""
import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.model.DataClass import PredictionResult, ModelConfig
from src.model.DummyDataGenerator import DummyDataGenerator
from src.model.ImagePreprocessor import ImagePreprocessor
from src.model.ModelDownloader import ModelDownloader
from src.model.ONNXPredictor import ONNXPredictor
from src.model.MNISTPipeline import MNISTPipeline

__all__ = [
    "PredictionResult",
    "ModelConfig",
    "DummyDataGenerator",
    "ImagePreprocessor",
    "ModelDownloader",
    "ONNXPredictor",
    "MNISTPipeline",
]
