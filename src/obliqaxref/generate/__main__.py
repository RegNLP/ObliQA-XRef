"""
python -m obliqaxref.generate

Module entrypoint for the generator package.
"""

from __future__ import annotations

from obliqaxref.generate.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
