import torch
import matplotlib.pyplot as plt

from models.cnn import CNN
from models.vit import ViT
from utils.dataset import get_cifar10_loaders


def main():

    # --------------------------------------------------
    # Select device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------
    # Load CIFAR-10 test image
    # --------------------------------------------------

    _, test_loader = get_cifar10_loaders(batch_size=1)

    images, labels = next(iter(test_loader))

    images = images.to(device)

    # --------------------------------------------------
    # Load trained models
    # --------------------------------------------------

    cnn = CNN(num_classes=10)

    vit = ViT(num_classes=10)

    cnn.load_state_dict(
        torch.load(
            "outputs/cnn_model.pth",
            map_location=device
        )
    )

    vit.load_state_dict(
        torch.load(
            "outputs/vit_model.pth",
            map_location=device
        )
    )

    cnn.to(device)
    vit.to(device)

    cnn.eval()
    vit.eval()

    # --------------------------------------------------
    # CNN representation
    # --------------------------------------------------

    with torch.no_grad():

        # Use CNN layers before AdaptiveAvgPool
        cnn_features = cnn.features[:8](images)

    # Shape:
    # [1, 128, 8, 8]

    cnn_feature_map = cnn_features[0]

    # Average across the 128 feature channels
    # Result: 8 x 8 spatial representation
    cnn_visualization = cnn_feature_map.mean(
        dim=0
    ).cpu()

    # --------------------------------------------------
    # ViT representation
    # --------------------------------------------------

    with torch.no_grad():

        # Convert image into 4x4 patches
        x = vit.patch_embedding(images)

        # Flatten spatial dimensions
        x = x.flatten(2)

        # Convert to sequence of patch tokens
        x = x.transpose(1, 2)

        # Add CLS token
        batch_size = x.size(0)

        cls_tokens = vit.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            (cls_tokens, x),
            dim=1
        )

        # Add positional embeddings
        x = x + vit.pos_embedding

        vit_tokens = x

    # Remove CLS token
    patch_tokens = vit_tokens[:, 1:, :]

    # Average embedding dimensions
    # 64 patches -> 8 x 8 grid
    vit_visualization = patch_tokens[0].mean(
        dim=1
    ).reshape(8, 8).cpu()

    # --------------------------------------------------
    # Prepare original image
    # --------------------------------------------------

    image = images[0].cpu()

    # CIFAR-10 normalization values
    mean = torch.tensor(
        [0.4914, 0.4822, 0.4465]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.2470, 0.2435, 0.2616]
    ).view(3, 1, 1)

    # Undo normalization
    image = image * std + mean

    # Keep values between 0 and 1
    image = torch.clamp(
        image,
        0,
        1
    )

    # --------------------------------------------------
    # Create comparison plot
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    # Original image
    axes[0].imshow(
    image.permute(1, 2, 0),
    interpolation="nearest"
)

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")

    # CNN representation
    axes[1].imshow(
    cnn_visualization,
    cmap="viridis",
    interpolation="nearest"
)

    axes[1].set_title(
        "CNN Feature Representation"
    )

    axes[1].axis("off")

    # ViT representation
    axes[2].imshow(
    vit_visualization,
    cmap="viridis",
    interpolation="nearest"
)

    axes[2].set_title(
        "ViT Patch Representation"
    )

    axes[2].axis("off")

    plt.tight_layout()

    # --------------------------------------------------
    # Save visualization
    # --------------------------------------------------

    plt.savefig(
        "outputs/plots/representation_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "Representation comparison saved."
    )


if __name__ == "__main__":
    main()