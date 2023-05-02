import torch
from torch import nn
import requests
import zipfile
from pathlib import Path


def download_data_small():
    data_path = Path("data/")
    image_path = data_path/"pizza_steak_sushi"

    if not image_path.is_dir():
        image_path.mkdir(parents=True, exist_ok=True)

    # Download zip folder with data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as filehandle:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")# noqa 5501
        filehandle.write(request.content)

    # Unzip folder
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        zip_ref.extractall(image_path)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    download_data_small()


if __name__ == "__main__":
    main()
