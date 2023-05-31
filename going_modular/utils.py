import torch
from pathlib import Path
import argparse
from typing import List
import torchvision
from tqdm.auto import tqdm
from timeit import default_timer as timer
from PIL import Image


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_save_path = target_dir_path / model_name

    torch.save(obj=model.state_dict(),
               f=model_save_path)


def set_seeds(seed: int = 42):

    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def parse_arguments():
    """
    Parse arguments:
    - batch size
    - hidden units
    - learning rate as lr
    - num of epochs
    """
    parser = argparse.ArgumentParser(description='Script train model TinyVGG')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and testing')
    parser.add_argument('--hidden_units', type=int, default=10,
                        help='Number of hidden units in the model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs for training')

    args = parser.parse_args()
    return args


def pred_and_store(paths: List[Path],
                   model: torch.nn.Module,
                   transform: torchvision.transforms,
                   class_names: List[str],
                   device="cuda"):

    pred_list = []
    for path in tqdm(paths):
        pred_dict = {}
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name

        start_time = timer()

        img = Image.open(path)

        transformed_img = transform(img).unsqueeze(0).to(device)

        model.to(device)
        model.eval()

        with torch.inference_mode():
            pred_logit = model(transformed_img)
            pred_prob = torch.softmax(input=pred_logit, dim=1)
            pred_label = torch.argmax(input=pred_prob, dim=1)
            pred_class = class_names[pred_label.cpu()]

            pred_dict["pred_prob"] = pred_prob
            pred_dict["pred_class"] = pred_class

            end_time = timer()
            pred_dict["prediction_time"] = round(end_time - start_time, 4)

        pred_dict["correct"] = class_name == pred_class

        pred_list.append(pred_dict)

    return pred_list