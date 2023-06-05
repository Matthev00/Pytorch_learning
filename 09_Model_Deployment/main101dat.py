import sys
sys.path.append("E:/projekty python/PyTorch course/going_modular")
sys.path.append("E:/projekty python/PyTorch course")
import ssl # noqa 5501
ssl._create_default_https_context = ssl._create_unverified_context # noqa 5501

import data_setup # noqa 5501
import model_builder # noqa 5501
import engine # noqa 5501
import utils # noqa 5501
from helper_functions import plot_loss_curves, create_writer, download_data, pred_and_plot_image # noqa 5501

import torch # noqa 5501
import torchvision # noqa 5501
from torch import nn # noqa 5501
from torchvision import transforms, datasets # noqa 5501data_20_percent_path
from torchinfo import summary # noqa 5501
from torchmetrics import ConfusionMatrix # noqa 5501
from torch.utils.data import DataLoader # noqa 5501
from torch.utils.tensorboard import SummaryWriter # noqa 5501

import random # noqa 5501
from typing import List, Tuple # noqa 5501
from PIL import Image # noqa 5501
from timeit import default_timer as timer # noqa 5501
import matplotlib.pyplot as plt # noqa 5501
from pathlib import Path # noqa 5501
import pandas as pd # noqa 5501
import gradio as gr # noqa 5501
import os # noqa 5501


def split_dataset(dataset: torchvision.datasets,
                  split: float = 0.2,
                  seed: int = 42):

    length1 = int(split * len(dataset))
    length2 = len(dataset) - length1

    utils.set_seeds(seed)

    random_split_1, random_split_2 = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[length1, length2],
        generator=torch.manual_seed(seed))

    return random_split_1, random_split_2


def main():
    args = utils.parse_arguments()
    BATCH_SIZE = args.batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = os.cpu_count()

    effnetb2, effnetb2_transforms = model_builder.create_effnetb2(
        out_features=101,
        device=device)

    # print(summary(effnetb2,
    #               input_size=(1, 3, 224, 224),
    #               col_names=["input_size", "output_size", "num_params", "trainable"], # noqa 5501
    #               col_width=20,
    #               row_settings=["var_names"]))

    train_effnetb2_transforms = transforms.Compose(
        [transforms.TrivialAugmentWide(),
         effnetb2_transforms])

    # print(effnetb2_transforms)

    data_dir = Path("data")
    data_path = data_dir / "Food101"
    train_data = datasets.Food101(root=data_path,
                                  split="train",
                                  transform=train_effnetb2_transforms,
                                  download=True)

    test_data = datasets.Food101(root=data_path,
                                 split="test",
                                 transform=effnetb2_transforms,
                                 download=True)

    train_data_food101_20_percent, _ = split_dataset(dataset=train_data,
                                                     split=0.2)

    # Create testing 20% split of Food101
    test_data_food101_20_percent, _ = split_dataset(dataset=test_data,
                                                    split=0.2)


    train_dataloader = DataLoader(dataset=train_data_food101_20_percent,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)

    test_dataloader = DataLoader(dataset=test_data_food101_20_percent,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS)

    # # Train model
    # # Setup optimizer
    # optimizer = torch.optim.Adam(params=effnetb2.parameters(),
    #                              lr=1e-3)

    # # Setup loss function
    # loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # # throw in a little label smoothing because so many classes

    # # Want to beat original Food101 paper with 20% of data
    # utils.set_seeds()
    # effnetb2_food101_results = engine.train(model=effnetb2,
    #                                         train_dataloader=train_dataloader,
    #                                         test_dataloader=test_dataloader,
    #                                         optimizer=optimizer,
    #                                         loss_fn=loss_fn,
    #                                         epochs=5,
    #                                         device=device)

    # # Load model
    effnetb2.load_state_dict(torch.load("effnetb2_food101.pth"))

    # effnetb2_size = Path("effnetb2_food101.pth").stat().st_size // (1024*1024)
    # print(train_data.classes[:10])

    # # Write Food101 class names list to file
    demo_path = Path("demos")
    class_path = demo_path / "foodvision_big" / "class_names.txt"

    with open(class_path, "w") as filehandle:
        filehandle.write("\n".join(train_data.classes))



if __name__ == "__main__":
    main()
