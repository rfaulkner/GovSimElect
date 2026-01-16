"""
LLM Judge Package

A flexible framework for using Large Language Models to classify and analyze text data
according to custom taxonomies.
"""

from .judge import LLMJudge
from .taxonomy import Taxonomy
from .processors import TextProcessor, BatchProcessor

__version__ = "1.0.0"
__all__ = ["LLMJudge", "Taxonomy", "TextProcessor", "BatchProcessor"]