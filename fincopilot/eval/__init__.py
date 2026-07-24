"""Evaluation harness: retrieval ablations against a hand-verified golden set."""

from .golden import GoldenQuestion, for_ticker
from .run import ABLATIONS, Configuration, evaluate, format_table

__all__ = [
    "GoldenQuestion",
    "for_ticker",
    "Configuration",
    "ABLATIONS",
    "evaluate",
    "format_table",
]
