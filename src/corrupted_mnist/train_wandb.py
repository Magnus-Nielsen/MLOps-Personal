#!
import os

import matplotlib.pyplot as plt
import torch
import typer
from dotenv import load_dotenv

import wandb
from corrupted_mnist.data import corrupt_mnist
from corrupted_mnist.model import ConvNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(
    lr: float = typer.Option(1e-3, "--lr", "-l", help="Learning rate for training"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for training"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
) -> None:
    """Train a model on MNIST."""

    load_dotenv()
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        config={"lr": lr, "batch-size": batch_size, "epochs": epochs},
    )

    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = ConvNet().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            wandb.log({"epoch": epoch + 1, "loss": loss.item(), "accuracy": accuracy})

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pt")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.tight_layout()

    # Save locally
    fig.savefig("reports/figures/training_statistics.png")

    # Log the plot to WandB
    wandb.log({"training_statistics": wandb.Image(fig)})

    plt.close(fig)

    artifact = wandb.Artifact(name="mnist_model", type="model")
    artifact.add_file(local_path="models/model.pt", name="model checkpoint")
    artifact.save()


if __name__ == "__main__":
    typer.run(train)
