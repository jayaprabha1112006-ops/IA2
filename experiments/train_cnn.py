import torch

from models.cnn import CNN
from utils.dataset import get_cifar10_loaders
from utils.train import train_model, evaluate_model


def main():

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Dataset
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=64
    )

    # Model
    model = CNN(num_classes=10)

    print("\nModel:")
    print(model)

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("\nTotal parameters:", total_parameters)

    # Training
    print("\nStarting CNN training...\n")

    history = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=5,
        learning_rate=0.001
    )

    # Evaluation
    print("\nEvaluating CNN...\n")

    test_accuracy, inference_time = evaluate_model(
        model,
        test_loader,
        device
    )

    # Save model
    torch.save(
        model.state_dict(),
        "outputs/cnn_model.pth"
    )

    print("\nCNN training completed.")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Inference time: {inference_time:.2f}s")


if __name__ == "__main__":
    main()