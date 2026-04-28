"""
ObliQA-XRef evaluation module.

Contains:
- ResourceStats: Intrinsic corpus/crossref/benchmark statistics
- split: Stratified train/test/dev splits by method/persona/ref_type
- HumanEval: Human evaluation CSV generation
"""

from obliqaxref.eval import HumanEval, ResourceStats

__all__ = ["ResourceStats", "HumanEval", "split_data"]
