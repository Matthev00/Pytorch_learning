import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
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
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
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
        self.cllasifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.cllasifier(x)
        return x


def download_data_small(data_path, image_path):

    if not image_path.is_dir():
        image_path.mkdir(parents=True, exist_ok=True)

    # Download zip folder with data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as filehandle:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")# noqa 5501
        filehandle.write(request.content)

    # Unzip folder
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        zip_ref.extractall(image_path)


def walk_throught_dir(dir_path):
    output = ""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        output += f'There are {len(dirnames)} dirs and {len(filenames)} images in {dir_path}.\n'# noqa 5501
    return output


def get_images_paths(image_path: Path,
                     patern: str):
    return list(image_path.glob(patern))


def plot_transformed_images(image_paths: list,
                            transforms, n=3, seed=None):
    """
    Selects random images and show original vs transformed ver.
    """
    if seed:
        random.seed(seed)
    random_images_paths = random.sample(image_paths, k=n)
    for image_path in random_images_paths:
        with Image.open(image_path) as im:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(im)
            ax[0].set_title(f'Original\nSize: {im.size}')
            ax[0].axis(False)

            # Transformed
            transformed_image = transforms(im).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f'Transformed\nShape: {transformed_image.shape}')
            ax[1].axis(False)

            fig.suptitle(f'Class: {image_path.parent.stem}', fontsize=16)

            plt.show()


def turn_into_dataset(train_dir: Path,
                      test_dir: Path,
                      train_transform: transforms,
                      test_transform: transforms):

    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transform,
                                      target_transform=None)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=test_transform)

    return train_data, test_data


def dispaly_random_img(dataset: torch.utils.data.Dataset,
                       classes: List[str] = None,
                       n: int = 10,
                       display_shape: bool = True,
                       seed: int = None):
    if n > 10:
        n = 10
        display_shape = False

    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize=(16, 8))

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample]
        targ_adjust_image = targ_image.permute(1, 2, 0)

        plt.subplot(1, n, i+1)
        plt.imshow(targ_adjust_image)
        plt.axis(False)
        if classes:
            title = f'Class: {classes[targ_label]}'
            if display_shape:
                title += f"\nShape: {targ_adjust_image.shape}"
        plt.title(title)

    plt.show()


def train_step(model: nn.Module,
               dataloader: torch.utils.data.dataloader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):

    model.train()

    train_loss, train_acc = 0, 0
    for batch, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)

        label_pred = model(image)

        loss = loss_fn(label_pred, label)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        label_pred_class = torch.argmax(torch.softmax(input=label_pred, dim=1), dim=1) # noqa 5501
        train_acc += (label_pred_class == label).sum().item()/len(label_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: nn.Module,
              dataloader: torch.utils.data.dataloader,
              loss_fn: torch.nn.Module,
              device):

    model.eval()

    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (image, label) in enumerate(dataloader):
            image, label = image.to(device), label.to(device)

            test_pred_logits = model(image)

            loss = loss_fn(test_pred_logits, label)
            test_loss += loss.item()

            test_label_pred_class = torch.argmax(torch.softmax(input=test_pred_logits, dim=1), dim=1) # noqa 5501
            test_acc += (test_label_pred_class == label).sum().item()/len(test_pred_logits) # noqa 5501

        test_acc /= len(dataloader)
        test_loss /= len(dataloader)

    return test_loss, test_acc


def train(model: nn.Module,
          train_dataloader: torch.utils.data.dataloader,
          test_dataloader: torch.utils.data.dataloader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          device="cuda",
          epochs: int = 5):

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def plot_loss_curves(results: Dict[str, List[float]]):
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="est_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_acc")
    plt.plot(epochs, test_accuracy, label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


def download_custom_image(path: Path):
    if not path.is_file():
        with open(path, "wb") as f:
            # When downloading from GitHub, need to use the "raw" file link
            request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg") # noqa 5501
            f.write(request.content)


def predict_on_image(path: Path,
                     model: nn.Module,
                     class_names: List,
                     device: torch.device = 'cuda'):

    custom_image_unit8 = torchvision.io.read_image(str(path))
    custom_image_unit8 = custom_image_unit8.type(torch.float32)/255

    resize_tranform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])
    custom_image = resize_tranform(custom_image_unit8).unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        pred_label = model(custom_image.to(device))

    pred_prob_label = torch.softmax(input=pred_label, dim=1)
    label = torch.argmax(input=pred_prob_label, dim=1)

    return class_names[label.item()]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    # download_data_small(data_path=data_path,
    #                     image_path=image_path)

    # print(walk_throught_dir(dir_path=image_path))

    # data_transform = transforms.Compose([
    #     transforms.Resize(size=(64, 64)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor()
    # ])

    simple_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    train_transform_trivial = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=32),
        transforms.ToTensor()
        ])

    # plot_transformed_images(image_paths=images_path_list,
    #                         transforms=data_transform,
    #                         n=3,
    #                         seed=42)

    train_data, test_data = turn_into_dataset(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_transform_trivial,
        test_transform=simple_transform)

    class_names = train_data.classes
    # class_dict = train_data_simple.class_to_idx

    # Set hyperparamiters
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()

    # Turn datasets into dataloaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS)

    # dispaly_random_img(dataset=train_data,
    #                    n=9,
    #                    classes=class_names,
    #                    display_shape=True,
    #                    seed=None)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    NUM_EPOCHS = 5
    model_0 = TinyVGG(input_shape=3,
                      hidden_units=10,
                      output_shape=len(class_names)).to(device)

    model_1 = TinyVGG(input_shape=3,
                      hidden_units=10,
                      output_shape=len(class_names)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # image_batch, label_batch = next(iter(train_dataloader_simple))
    # y = model_0(image_batch.to(device))
    # print(y)
    # print(summary(model=model_0, input_size=[1, 3, 64, 64]))

    # start_time = timer()

    # model_1_results = train(model=model_1,
    #                         train_dataloader=train_dataloader,
    #                         test_dataloader=test_dataloader,
    #                         optimizer=optimizer,
    #                         loss_fn=loss_fn,
    #                         epochs=NUM_EPOCHS,
    #                         device=device)
    # end_time = timer()
    # print(f'Time: {end_time-start_time:.3f}')
    # plot_loss_curves(model_1_results)
    # print(model_1_results)

    # Save model
    model_path = Path('models')
    model_name = '04_pytorch_custom_datasets2.pth'
    model_save_path = model_path / model_name

    torch.save(obj=model_1.state_dict(), f=model_save_path)

    # Load the model
    model_1.load_state_dict(torch.load(f=model_save_path))
    model_1 = torch.compi

    # Predict on custom data
    custom_image_path = data_path / "04-pizza-dad.jpeg"
    download_custom_image(custom_image_path)

    label = predict_on_image(path=custom_image_path,
                             model=model_1,
                             class_names=class_names)

    print(label)


if __name__ == "__main__":
    main()
