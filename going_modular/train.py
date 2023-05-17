import torch
import data_setup
import model_builder
import engine
import utils
from torchvision import transforms


NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001


def main():
    # Setup dir
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

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

    utils.save_model(model=model,
                     target_dir="models",
                     model_name="tinyVGG_model_5_epochs.pth")


if __name__ == "__main__":
    main()
