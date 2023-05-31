import torch
from torch import nn
import utils
import torchvision


class TinyVGG(nn.Module):
    """
    Creates TinyVGG model

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch. # noqa 5501

    Args:
        input_shape: An integer indicating number of input channels.
        hidden_units: An integer indicating number of hidden units between layers.
        output_shape: An integer indicating number of output units.
    """

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def create_effnetb0(out_features,
                    device):
    effnetb0_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=effnetb0_weights).to(device) # noqa 5501

    for param in model.features.parameters():
        param.requires_grad = False

    utils.set_seeds(42)

    # # Set cllasifier to suit problem
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280,
                  out_features=out_features,
                  bias=True).to(device))

    model.name = "effnetb0"
    return model


def create_effnetb2(out_features,
                    device):
    effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=effnetb2_weights).to(device) # noqa 5501

    for param in model.features.parameters():
        param.requires_grad = False

    utils.set_seeds(42)

    # # Set cllasifier to suit problem
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1408,
                  out_features=out_features,
                  bias=True).to(device))

    model.name = "effnetb2"
    return model
