import re

import pytest
import torch

from corrupted_mnist.model import ConvNet  # Assume this is the file name containing the ConvNet class


def test_model():
    model = ConvNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)


def test_dropout_behavior():
    model = ConvNet()
    model.eval()  # Disable dropout during evaluation
    dummy_input = torch.randn(1, 1, 28, 28)

    # Check output consistency
    output_1 = model(dummy_input)
    output_2 = model(dummy_input)

    assert torch.equal(output_1, output_2), "Dropout should not affect evaluation mode outputs."

    model.train()  # Enable dropout during training
    output_3 = model(dummy_input)
    output_4 = model(dummy_input)

    assert not torch.equal(output_3, output_4), "Dropout should create different outputs during training mode."


def test_error_on_wrong_shape():
    model = ConvNet()
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
    with pytest.raises(ValueError, match=re.escape("Expected each sample to have shape [1, 28, 28]")):
        model(torch.randn(1, 1, 28, 29))
