#!/usr/bin/env python
"""Analyze training metrics history and auto-diagnose learning behavior."""

import argparse
import json
from pathlib import Path
import torch
from src.training.checkpoint import load_checkpoint
from src.tools.learning_curves import LearningCurveAnalyzer
from src.tools.reporting import ReportGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Path to checkpoint or metrics_history JSON")
    parser.add_argument("--output", type=str, default="reports/learning_curves")
    parser.add_argument("--overfit-window", type=int, default=10)
    parser.add_argument("--plateau-window", type=int, default=15)
    parser.add_argument("--divergence-ratio", type=float, default=3.0)
    args = parser.parse_args()

    if args.ckpt_path.endswith(".json"):
        with open(args.ckpt_path) as f:
            data = json.load(f)
            metrics_history = data if isinstance(data, list) else data.get("metrics_history", [])
    else:
        ckpt = load_checkpoint(args.ckpt_path)
        metrics_history = ckpt.get("metrics_history", [])

    if not metrics_history:
        print("No metrics_history found.")
        return

    analyzer = LearningCurveAnalyzer(
        overfit_window=args.overfit_window,
        plateau_window=args.plateau_window,
        divergence_ratio=args.divergence_ratio,
    )
    diagnosis = analyzer.analyze(metrics_history)
    summary = analyzer.generate_summary(diagnosis)
    print("\n" + summary)

    output_dir = Path(args.output)
    reporter = ReportGenerator(output_dir=output_dir)
    report_path = reporter.generate_learning_curve_report(diagnosis, metrics_history, output_dir)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
