from pathlib import Path
from typing import Annotated

import joblib
import typer
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = typer.Typer()
train_app = typer.Typer()
app.add_typer(train_app, name="train")

# Load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


@train_app.command()
def svm(
    output: Annotated[str, typer.Option("--output", "-o", help="Path to save the trained model")] = "model.ckpt",
    kernel: str = "linear",
):
    """Train and evaluate the model."""

    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel=kernel, random_state=42)
    model.fit(x_train, y_train)

    # Save the model
    save_path = Path("models") / output
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")


@train_app.command()
def knn(
    output: Annotated[str, typer.Option("--output", "-o", help="Path to save the trained model")] = "model.ckpt",
    k: int = 5,
):
    """Train and evaluate the model."""

    # Train a Support Vector Machine (SVM) model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    # Save the model
    save_path = Path("models") / output
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")


@app.command()
def evaluate(model_path: Annotated[str, typer.Argument(help="Path to the saved model")]):
    """Load a trained model and evaluate it on the test set."""

    # Load the model
    model_path = Path("models") / model_path
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        raise typer.Exit(code=1)

    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Make predictions
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    app()
