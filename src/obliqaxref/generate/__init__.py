# src/obliqaxref/generate/__init__.py

"""
obliqaxref.generate

Generator module for ObliQA-XRef:
- DPEL generation (baseline): obliqaxref.generate.dpel
- SCHEMA extraction + generation: obliqaxref.generate.schema
- Statistics and paper tables: obliqaxref.generate.stats

This package is intentionally import-light to keep CLI startup fast.
"""

__all__ = ["common", "dpel", "schema", "stats"]
