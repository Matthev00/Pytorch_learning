import torch
import data_setup
import model_builder
import engine
import utils
from torchvision import transforms
from pathlib import Path
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script train model TinyVGG')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and testing')
    parser.add_argument('--hidden_units', type=int, default=10,
                        help='Number of hidden units in the model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs for training')

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_arguments()
    HIDDEN_UNITS = args.hidden_units
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    # Setup dir
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create transform
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Create dataloaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, # noqa 5501
                                                                                test_dir=test_dir, # noqa 5501
                                                                                transform=data_transform, # noqa 5501
                                                                                batch_size=BATCH_SIZE) # noqa 5501

    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 optimizer=optimizer,
                 loss_fn=loss_fn,
                 epochs=NUM_EPOCHS,
                 device=device)

    # utils.save_model(model=model,
    #                  target_dir="../models",
    #                  model_name="tinyVGG_model_5_epochs.pth")

    model.load_state_dict(torch.load("models/tinyVGG_model_5_epochs.pth"))


if __name__ == "__main__":
    main()
