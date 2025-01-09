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

    run = wandb.init()

    downloaded_model_path = run.use_model(
        name="advanced-deep-learning-course-org/wandb-registry-model/MLOPS Corrupted MNIST registry:v0"
    )
    model = ConvNet()
    model.load_state_dict(torch.load(downloaded_model_path))
    model.eval()

    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    statistics = {"train_loss": [], "train_accuracy": []}
    with torch.no_grad():
        for epoch in range(1):
            for i, (img, target) in enumerate(train_dataloader):
                img, target = img.to(DEVICE), target.to(DEVICE)
                y_pred = model(img)

                accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                statistics["train_accuracy"].append(accuracy)

    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    axs.plot(statistics["train_accuracy"])
    axs.set_title("Train accuracy")
    fig.tight_layout()

    # Save locally
    fig.savefig("reports/figures/training_statistics_artifact.png")


if __name__ == "__main__":
    typer.run(train)
