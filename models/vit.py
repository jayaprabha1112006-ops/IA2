import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()

        # Number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Convert image patches into embeddings
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Class token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Linear(
            embed_dim,
            num_classes
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x: [batch, 3, 32, 32]

        # Convert image into patch embeddings
        x = self.patch_embedding(x)

        # [batch, embed_dim, 8, 8]
        x = x.flatten(2)

        # [batch, embed_dim, 64]
        x = x.transpose(1, 2)

        # [batch, 64, embed_dim]

        batch_size = x.size(0)

        # Add class token
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat(
            (cls_tokens, x),
            dim=1
        )

        # Add positional information
        x = x + self.pos_embedding

        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Take class token
        x = x[:, 0]

        # Classification
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = ViT()

    print(model)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nTotal parameters:", total_parameters)

    # Test with a fake CIFAR-10 batch
    x = torch.randn(64, 3, 32, 32)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)