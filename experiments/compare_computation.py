import matplotlib.pyplot as plt


def main():

    models = ["CNN", "ViT"]

    parameters = [94538, 809098]
    inference_time = [7.10, 15.78]

    # -----------------------------
    # Parameter comparison
    # -----------------------------

    plt.figure(figsize=(7, 5))

    bars = plt.bar(models, parameters)

    plt.ylabel("Number of Parameters")
    plt.xlabel("Model")
    plt.title("CNN vs ViT Parameter Count")

    for bar, value in zip(bars, parameters):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/parameter_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # -----------------------------
    # Inference time comparison
    # -----------------------------

    plt.figure(figsize=(7, 5))

    bars = plt.bar(models, inference_time)

    plt.ylabel("Inference Time (seconds)")
    plt.xlabel("Model")
    plt.title("CNN vs ViT Inference Time")

    for bar, value in zip(bars, inference_time):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}s",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/inference_time_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Computational comparison saved.")


if __name__ == "__main__":
    main()