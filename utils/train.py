import time

import torch
import torch.nn as nn


def train_model(model, train_loader, device, epochs=5, learning_rate=0.001):
    """
    Train a classification model and return training history.
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.to(device)

    history = {
        "train_loss": [],
        "train_accuracy": []
    }

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = 100 * correct / total

        epoch_time = time.time() - start_time

        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.2f}% "
            f"Time: {epoch_time:.2f}s"
        )

    return history


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.
    """

    model.eval()

    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    inference_time = time.time() - start_time

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test inference time: {inference_time:.2f}s")

    return accuracy, inference_time