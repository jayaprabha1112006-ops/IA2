import matplotlib.pyplot as plt


def main():

    models = ["CNN", "ViT"]
    accuracies = [64.98, 66.72]

    plt.figure(figsize=(7, 5))

    bars = plt.bar(models, accuracies)

    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Model")
    plt.title("CNN vs ViT Classification Performance")

    plt.ylim(0, 100)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{accuracy:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/classification_performance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Classification performance comparison saved.")


if __name__ == "__main__":
    main()