import torch
from pathlib import Path
import argparse


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
