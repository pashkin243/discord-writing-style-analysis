import numpy as np
import matplotlib.pyplot as plt

uniqueness = {
    # data
}

precision = {
    # data
}

recall = {
    # data
}

f1 = {
    # data
}


def analyze_metric(metric_name, values):
    users = list(uniqueness.keys())

    x = np.array([uniqueness[u] for u in users])
    y = np.array([values[u] for u in users])

    corr = np.corrcoef(x, y)[0, 1]
    print(f"\nCorrelation (uniqueness vs {metric_name}): {corr:.4f}")

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y)

    for i, u in enumerate(users):
        plt.text(x[i], y[i], u, fontsize=8)

    plt.xlabel("Style Uniqueness")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"Uniqueness vs {metric_name.capitalize()}")
    plt.grid(True)

    filename = f"uniqueness_vs_{metric_name}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

    print(f"Saved plot to {filename}")


analyze_metric("recall", recall)
analyze_metric("precision", precision)
analyze_metric("f1", f1)