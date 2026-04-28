"""
Entrypoint for ObliQA-XRef evaluation CLI.

Usage:
    python -m obliqaxref.eval.run <subcommand> [options]
    or
    python src/obliqaxref/eval/run.py <subcommand> [options]

Subcommands:
    split      Stratified train/test/dev split
    humaneval  Generate combined human evaluation CSV
    finalize   Generate final datasets split by method

See --help for details.
"""

import sys

from obliqaxref.eval.cli import main

if __name__ == "__main__":
    sys.exit(main())
