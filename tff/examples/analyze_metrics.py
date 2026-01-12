"""Example: Analyze training metrics.

This script demonstrates how to load and analyze metrics from a training run.
"""

from pathlib import Path
from tff.metrics import load_metrics, load_training_info, print_training_summary


def analyze_training_run(checkpoint_dir: str = "checkpoints") -> None:
    """Analyze a training run's metrics.

    Args:
        checkpoint_dir: Directory containing training metrics
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_path}")
        return

    # Print summary
    print_training_summary(checkpoint_path)

    # Load detailed metrics
    print("\n" + "=" * 80)
    print("Loading detailed metrics...")
    print("=" * 80)

    try:
        metrics = load_metrics(checkpoint_path)
        print(f"\nLoaded {len(metrics)} step metrics")

        # Show first few steps
        print("\nFirst 5 steps:")
        print("-" * 80)
        print(f"{'Step':<8} {'Loss':<10} {'BPC':<10} {'Tokens':<15} {'Val BPC':<10}")
        print("-" * 80)
        for m in metrics[:5]:
            val_bpc = f"{m.val_bpc:.4f}" if m.val_bpc is not None else "N/A"
            print(
                f"{m.step:<8} {m.loss:<10.4f} {m.bpc:<10.4f} "
                f"{m.total_tokens_seen:<15,} {val_bpc:<10}"
            )

        # Show validation steps
        val_metrics = [m for m in metrics if m.val_loss is not None]
        if val_metrics:
            print(f"\nValidation steps ({len(val_metrics)} total):")
            print("-" * 80)
            print(f"{'Step':<8} {'Val Loss':<12} {'Val BPC':<10}")
            print("-" * 80)
            for m in val_metrics:
                print(f"{m.step:<8} {m.val_loss:<12.4f} {m.val_bpc:<10.4f}")

        # Analyze throughput
        if len(metrics) > 1:
            print("\n" + "=" * 80)
            print("Throughput Analysis")
            print("=" * 80)

            final_metric = metrics[-1]
            print(f"\nFinal throughput:")
            print(f"  Tokens/second: {final_metric.tokens_per_second:,.0f}")
            print(f"  Steps/second:  {final_metric.steps_per_second:.2f}")

            # Compute average over last 100 steps
            if len(metrics) >= 100:
                recent_metrics = metrics[-100:]
                avg_tokens_per_sec = sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics)
                avg_steps_per_sec = sum(m.steps_per_second for m in recent_metrics) / len(recent_metrics)

                print(f"\nAverage over last 100 logged steps:")
                print(f"  Tokens/second: {avg_tokens_per_sec:,.0f}")
                print(f"  Steps/second:  {avg_steps_per_sec:.2f}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure training has completed and metrics were saved.")


def compare_runs(dir1: str, dir2: str) -> None:
    """Compare metrics from two training runs.

    Args:
        dir1: First checkpoint directory
        dir2: Second checkpoint directory
    """
    print("=" * 80)
    print("Comparing Training Runs")
    print("=" * 80)

    try:
        info1 = load_training_info(dir1)
        info2 = load_training_info(dir2)

        print(f"\nRun 1: {dir1}")
        print(f"  Model params:  {info1.model_params:,}")
        print(f"  Total tokens:  {info1.total_tokens:,} ({info1.total_tokens / 1e6:.2f}M)")
        print(f"  Best val BPC:  {info1.best_val_bpc:.4f} (step {info1.best_val_step})")
        print(f"  Final val BPC: {info1.final_val_bpc:.4f}")

        print(f"\nRun 2: {dir2}")
        print(f"  Model params:  {info2.model_params:,}")
        print(f"  Total tokens:  {info2.total_tokens:,} ({info2.total_tokens / 1e6:.2f}M)")
        print(f"  Best val BPC:  {info2.best_val_bpc:.4f} (step {info2.best_val_step})")
        print(f"  Final val BPC: {info2.final_val_bpc:.4f}")

        print("\n" + "=" * 80)
        print("Comparison")
        print("=" * 80)

        bpc_diff = info2.best_val_bpc - info1.best_val_bpc
        better = "Run 2" if bpc_diff < 0 else "Run 1"
        print(f"\nBest validation BPC difference: {bpc_diff:+.4f}")
        print(f"Better model: {better}")

        efficiency1 = info1.total_tokens / info1.total_time_seconds
        efficiency2 = info2.total_tokens / info2.total_time_seconds
        print(f"\nThroughput comparison:")
        print(f"  Run 1: {efficiency1:,.0f} tokens/sec")
        print(f"  Run 2: {efficiency2:,.0f} tokens/sec")
        print(f"  Difference: {(efficiency2 - efficiency1) / efficiency1 * 100:+.1f}%")

    except FileNotFoundError as e:
        print(f"\nError: {e}")


def export_metrics_csv(checkpoint_dir: str, output_file: str = "metrics.csv") -> None:
    """Export metrics to CSV for external analysis.

    Args:
        checkpoint_dir: Directory containing training metrics
        output_file: Output CSV filename
    """
    import csv

    metrics = load_metrics(checkpoint_dir)

    output_path = Path(checkpoint_dir) / output_file

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'step', 'loss', 'bpc', 'learning_rate',
                'tokens_in_batch', 'total_tokens_seen',
                'elapsed_seconds', 'steps_per_second', 'tokens_per_second',
                'val_loss', 'val_bpc'
            ]
        )
        writer.writeheader()

        for m in metrics:
            writer.writerow(m.to_dict())

    print(f"Exported {len(metrics)} metrics to: {output_path}")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze_metrics.py <checkpoint_dir>")
        print("  python analyze_metrics.py compare <dir1> <dir2>")
        print("  python analyze_metrics.py export <checkpoint_dir> [output.csv]")
        print("\nExamples:")
        print("  python analyze_metrics.py checkpoints")
        print("  python analyze_metrics.py checkpoints/toy")
        print("  python analyze_metrics.py compare checkpoints/run1 checkpoints/run2")
        print("  python analyze_metrics.py export checkpoints metrics.csv")
        return

    command = sys.argv[1]

    if command == "compare":
        if len(sys.argv) < 4:
            print("Error: compare requires two checkpoint directories")
            return
        compare_runs(sys.argv[2], sys.argv[3])

    elif command == "export":
        if len(sys.argv) < 3:
            print("Error: export requires a checkpoint directory")
            return
        output = sys.argv[3] if len(sys.argv) > 3 else "metrics.csv"
        export_metrics_csv(sys.argv[2], output)

    else:
        # Assume it's a checkpoint directory
        analyze_training_run(command)


if __name__ == "__main__":
    main()
