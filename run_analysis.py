from __future__ import annotations

import argparse
from pathlib import Path

from pokebench30.evaluation.analysis import analyze_run


def main():
    parser = argparse.ArgumentParser(description="Analyze an existing PokéBench30 v0.2 run.")
    parser.add_argument("run_dir", type=Path, help="Example: results/run_20260927_190000")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    rows = analyze_run(args.run_dir, make_plots=not args.no_plots)
    print(f"Generated agent_summary.csv for {len(rows)} generation-agent groups in {args.run_dir}")


if __name__ == "__main__":
    main()
