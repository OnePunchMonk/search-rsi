"""Runs the full search-use-case benchmark (all three RSI levels, with control arms)
and writes reports/benchmark.md + reports/benchmark.json.

Usage: python experiments/run_search_benchmark.py [--iterations 40] [--seeds 0 1 2]
Equivalent to: search-rsi bench --out reports/
"""
from __future__ import annotations

import sys
from pathlib import Path

from search_rsi.cli import main

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "reports"
    main(["bench", "--out", str(out), *sys.argv[1:]])
