import os.path

import pytest
import torch

from corrupted_mnist.data import corrupt_mnist, normalize
from tests import _PATH_DATA


def test_normalize():
    # Create a sample tensor representing a batch of images
    images = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    # Call the normalize function
    normalized_images = normalize(images)

    # Assert that the mean of the normalized images is close to 0
    assert torch.isclose(normalized_images.mean(), torch.tensor(0.0), atol=1e-5), (
        "Mean is not close to 0 after normalization."
    )

    # Assert that the standard deviation of the normalized images is close to 1
    assert torch.isclose(normalized_images.std(), torch.tensor(1.0), atol=1e-5), (
        "Standard deviation is not close to 1 after normalization."
    )


processed_data_exists = all(
    os.path.exists(f"{_PATH_DATA}/processed/{filename}")
    for filename in ["train_images.pt", "train_target.pt", "test_images.pt", "test_target.pt"]
)


@pytest.mark.skipif(not processed_data_exists, reason="Data files not found")
def test_data():
    # Load the train and test datasets using the corrupt_mnist function
    train_set, test_set = corrupt_mnist()

    # Assert dataset lengths
    assert len(train_set) == 30_000, f"Expected 60000 training samples, but got {len(train_set)}"
    assert len(test_set) == 5_000, f"Expected 10000 test samples, but got {len(test_set)}"

    # Check sample shapes
    for dataset in [train_set, test_set]:
        for image, label in dataset:
            assert image.shape == (1, 28, 28), f"Expected image shape (1, 28, 28), but got {image.shape}"

    # Assert that all labels from 0 to 9 are represented in the train and test sets
    train_labels = [label.item() for _, label in train_set]
    test_labels = [label.item() for _, label in test_set]

    assert set(train_labels) == set(range(10)), (
        f"Not all labels (0-9) are represented in the training set. Found: {set(train_labels)}"
    )
    assert set(test_labels) == set(range(10)), (
        f"Not all labels (0-9) are represented in the test set. Found: {set(test_labels)}"
    )
