import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = Path('<path-to-saved-results>')


def format_seconds(seconds: float) -> str:
    minutes = seconds / 60
    return f"{minutes:.2f} minutes" if minutes >= 1 else f"{seconds:.2f} seconds"


def print_metrics(data: dict) -> None:
    metrics = data.get("metrics", {})
    confusion = data.get("confusion_matrix", [])
    confident = data.get("confident_errors", {})
    complexity = data.get("model_complexity", {})

    print("\n=== Evaluation Summary ===")

    print("\nMetrics:")
    print(f"  Accuracy          : {metrics.get('accuracy', 0) * 100:.2f}%")
    print(f"  Validation Loss   : {metrics.get('val_loss', 0):.4f}")
    print(f"  Validation Time   : {format_seconds(metrics.get('vali_time_sec', 0))}")
    print(f"  Avg Inference Time: {metrics.get('vali_inf_time_sec', 0):.4f} seconds")
    print(f"  Inference Rate    : {metrics.get('vali_fps', 0):.2f} frames/sec")

    print("\nConfident Errors:")
    print(f"  Confident False Real: {confident.get('CFR', 0)}")
    print(f"  Confident False Fake: {confident.get('CFF', 0)}")

    print("\nModel Complexity:")
    print(f"  Parameters: {complexity.get('params_millions', 0):.2f}M")
    print(f"  GFLOPs    : {complexity.get('gflops', 0):.2f}")


def plot_confusion_matrix(confusion, class_names=None):
    if not confusion:
        print("No confusion matrix data available to plot.")
        return

    matrix = np.array(confusion)
    num_classes = matrix.shape[0]
    if class_names is None or len(class_names) != num_classes:
        class_names = [f"Real" if i == 0 else "Fake" for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")

    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    cell_fontsize = 12

    ax.set_title("Confusion Matrix", fontsize=title_fontsize)
    ax.set_xlabel("Predicted", fontsize=label_fontsize)
    ax.set_ylabel("Actual", fontsize=label_fontsize)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, fontsize=tick_fontsize)
    ax.set_yticklabels(class_names, fontsize=tick_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=cell_fontsize,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    plt.tight_layout()
    plt.show()


def main():
    if not RESULTS_PATH.exists():
        print(f"No validation results found at {RESULTS_PATH.resolve()}")
        return

    with RESULTS_PATH.open('r', encoding='utf-8') as f:
        data = json.load(f)

    print_metrics(data)
    plot_confusion_matrix(
        data.get("confusion_matrix"),
        data.get("class_names")
    )


if __name__ == "__main__":
    main()
