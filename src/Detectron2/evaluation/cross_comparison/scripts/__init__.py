"""
Cross-comparison evaluation scripts package

Contains modular components for comparing failed images across RGB, Depth, and RGD models.
"""

from .tumor_evaluator import TumorEvaluator
from .json_utils import JSONHandler
from .image_processor import ImageProcessor
from .plotter import ComparisonPlotter
from .output_writer import OutputWriter

__all__ = [
    "TumorEvaluator",
    "JSONHandler",
    "ImageProcessor",
    "ComparisonPlotter",
    "OutputWriter",
]
