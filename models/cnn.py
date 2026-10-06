import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = CNN()

    print(model)

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("\nTotal parameters:", total_parameters)

    # Test with a fake CIFAR-10 batch
    x = torch.randn(64, 3, 32, 32)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)