import sys
sys.path.append("E:/projekty python/PyTorch course/going_modular")
sys.path.append("E:/projekty python/PyTorch course")

import data_setup # noqa 5501
import model_builder # noqa 5501
import engine # noqa 5501
import utils # noqa 5501
from helper_functions import plot_loss_curves, create_writer, download_data, pred_and_plot_image # noqa 5501

import torch # noqa 5501
import torchvision # noqa 5501
from torch import nn # noqa 5501
from torchvision import transforms, datasets # noqa 5501
from torchinfo import summary # noqa 5501
from torchmetrics import ConfusionMatrix # noqa 5501
from torch.utils.data import DataLoader # noqa 5501
from torch.utils.tensorboard import SummaryWriter # noqa 5501

import random # noqa 5501
from typing import List, Tuple # noqa 5501
from PIL import Image # noqa 5501
from timeit import default_timer as timer # noqa 5501
import matplotlib.pyplot as plt # noqa 5501
import argparse # noqa 5501
from pathlib import Path # noqa 5501


def pred_and_plot_image(model: nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device = "cuda"):
    # Open an image
    img = Image.open(image_path)

    # Create transformation
    if transform:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    model.to(device)

    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        image_pred = model(transformed_image.to(device))

    img_pred_probs = torch.softmax(input=image_pred, dim=1)
    img_label = torch.argmax(input=img_pred_probs, dim=1)
    plt.figure()
    plt.imshow(img)
    plt.title(f'Pred: {class_names[img_label]} | Prob: {img_pred_probs.max():.3f}') # noqa 5501
    plt.axis(False)
    plt.show()


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


def main():

    args = utils.parse_arguments()
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.set_seeds(42)

    # # Download data
    data_10_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", # noqa 5501
                                         destination="pizza_steak_sushi")  # noqa 5501

    data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip", # noqa 5501
                                         destination="pizza_steak_sushi_20_percent") # noqa 5501

    # # Setup data dir
    train_dir_10_percent = data_10_percent_path / "train"
    train_dir_20_percent = data_20_percent_path / "train"
    test_dir = data_10_percent_path / "test"

    # # Setup manual transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])

    # # Create data loaders
    train_dataloader_10_percent, test_dataloader, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir_10_percent, # noqa 5501
        test_dir=test_dir, # noqa 5501
        transform=simple_transform, # noqa 5501
        batch_size=BATCH_SIZE )# noqa 5501

    train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir_20_percent, # noqa 5501
        test_dir=test_dir, # noqa 5501
        transform=simple_transform, # noqa 5501
        batch_size=BATCH_SIZE )# noqa 5501

    # # # Get a summary of the model
    # print(summary(effnetb2,
    #               input_size=(32, 3, 224, 224),
    #               verbose=0,
    #               col_names=["input_size", "output_size", "num_params", "trainable"], # noqa 5501
    #               col_width=20,
    #               row_settings=["var_names"]))

    # # # Create experiment cases
    # epochs = [5, 10]
    # epochs = [10]
    # models = ["effnetb0", "effnetb2"]
    # models = ["effnetb2"]
    # train_dataloaders = {"data_10_percent": train_dataloader_10_percent,
    #                      "data_20_precent": train_dataloader_20_percent}
    # train_dataloaders = {"data_20_precent": train_dataloader_20_percent}

    # # # Experiment
    # utils.set_seeds(42)
    # OUT_FEATURES = len(class_names)

    # experiment_number = 0
    # for dataloader_name, train_dataloader in train_dataloaders.items():
    #     for epoch in epochs:
    #         for model_name in models:
    #             experiment_number += 1
    #             print(f"[INFO] Experiment number: {experiment_number}")
    #             print(f"[INFO] Model: {model_name}")
    #             print(f"[INFO] DataLoader: {dataloader_name}")
    #             print(f"[INFO] Number of epochs: {epoch}")
    #             if model_name == "effnetb0":
    #                 model = create_effnetb0(out_features=OUT_FEATURES,
    #                                         device=device)
    #             else:
    #                 model = create_effnetb2(out_features=OUT_FEATURES,
    #                                         device=device)

    #             loss_fn = nn.CrossEntropyLoss()
    #             optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # noqa 5501

    #             engine.train(model=model,
    #                          train_dataloader=train_dataloader,
    #                          test_dataloader=test_dataloader,
    #                          optimizer=optimizer,
    #                          loss_fn=loss_fn,
    #                          epochs=epoch,
    #                          device=device,
    #                          writer=create_writer(experiment_name=dataloader_name, # noqa 5501
    #                                               model_name=model_name,
    #                                               extra=f'{epoch}_epochs'))

    #             # # Save model
    #             save_filepath = f"07_{model_name}_{dataloader_name}_{epoch}_epochs.pth" # noqa 5501
    #             utils.save_model(model=model,
    #                              target_dir="../models",
    #                              model_name=save_filepath)
    #             print("-"*50 + "\n")

    # # Import best model
    best_model_path = "../models/07_effnetb2_data_20_precent_10_epochs.pth"
    best_model = create_effnetb2(out_features=3, device=device)
    best_model.load_state_dict(torch.load(best_model_path))
    effnetb2_model_size = Path(best_model_path).stat().st_size // (1024*1024)
    print(f"Model size {effnetb2_model_size}")

    # # Predict on data

    img_to_plot = 3
    test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
    img_path_sample = random.sample(population=test_image_path_list,
                                    k=img_to_plot)

    for img_path in img_path_sample:
        pred_and_plot_image(model=best_model,
                            image_path=img_path,
                            class_names=class_names,
                            image_size=(224, 224))

    custom_img_path = "../data/04-pizza-dad.jpeg"
    pred_and_plot_image(model=best_model,
                        image_path=custom_img_path,
                        class_names=class_names,
                        image_size=(224, 224))


if __name__ == "__main__":
    main()
