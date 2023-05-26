import sys
sys.path.append("E:/projekty python/PyTorch course/going_modular")
sys.path.append("E:/projekty python/PyTorch course")
import matplotlib.pyplot as plt # noqa 5501
import torch # noqa 5501
import torchvision # noqa 5501
import data_setup # noqa 5501
import model_builder # noqa 5501
import engine # noqa 5501
import utils # noqa 5501
from torch import nn # noqa 5501
from torchvision import transforms, datasets # noqa 5501
import argparse # noqa 5501
from pathlib import Path # noqa 5501
from torchinfo import summary # noqa 5501
from timeit import default_timer as timer # noqa 5501
from helper_functions import plot_loss_curves # noqa 5501
from typing import List, Tuple # noqa 5501
from PIL import Image # noqa 5501
import random # noqa 5501
from torchmetrics import ConfusionMatrix # noqa 5501
from mlxtend.plotting import plot_confusion_matrix # noqa 5501
from torch.utils.data import DataLoader # noqa 5501


def make_and_plot_confusion_matrix(model: nn.Module,
                                   dataloader: DataLoader,
                                   test_data,
                                   class_names: List[str],
                                   device: torch.device = "cuda"):

    y_preds = make_predictions_dataloader(model=model,
                                          data=dataloader,
                                          device=device)

    return y_preds, test_data.targets
    # # Set up confusion instance and compere oredictions to targets
    # conf_mat = ConfusionMatrix(num_classes=len(class_names),
    #                            task="multiclass")
    # conf_mat_tensor = conf_mat(preds=y_preds.item(),
    #                            target=test_data.targets)

    # fig, ax = plot_confusion_matrix(conf_mat=conf_mat_tensor.numpy(),
    #                                 class_names=class_names,
    #                                 figsize=(10, 7))
    # plt.show()


def make_predictions_dataloader(model: torch.nn.Module,
                                data: DataLoader,
                                device: torch.device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for image, label in data:
            image, label = image.to(device), label.to(device)

            pred_logit = model(image)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0).argmax(dim=1)# noqa 5501

            pred_probs.append(pred_prob.to("cpu"))
    return torch.cat(pred_probs)


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


def parse_arguments():
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


def main():
    args = parse_arguments()
    HIDDEN_UNITS = args.hidden_units
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    # Setup data dir
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Create a transforms pipeline manually (required for torchvision < 0.13)
    # manual_transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])

    # # Get a set of pretrained model weights(auto creation)
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    auto_transform = weights.transforms()

    # # Create dataloaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, # noqa 5501
                                                                                   test_dir=test_dir, # noqa 5501
                                                                                   transform=auto_transform, # noqa 5501
                                                                                   batch_size=BATCH_SIZE) # noqa 5501

    # # Setup the model with pretrained weights send it to the target device
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # # Frezee all base layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names
    output_shape = len(class_names)

    # # Recreate classifier layer to fit in our case
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280,
                  out_features=output_shape,
                  bias=True)).to(device)

    # print(summary(model=model,
    #               input_size=(32, 3, 224, 224),  # make sure this is "input_size", not "input_shape" # noqa 5501
    #               # col_names=["input_size"], # uncomment for smaller output
    #               col_names=["input_size", "output_size", "num_params", "trainable"], # noqa 5501
    #               col_width=20,
    #               row_settings=["var_names"]))

    # # # Create Loss function and optimizer
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # # # Train model
    # start_time = timer()
    # result = engine.train(model=model,
    #                       train_dataloader=train_dataloader,
    #                       test_dataloader=test_dataloader,
    #                       optimizer=optimizer,
    #                       loss_fn=loss_fn,
    #                       epochs=NUM_EPOCHS,
    #                       device=device)
    # end_time = timer()
    # print(f'Training time: {end_time-start_time:.3f}')
    # print(result)

    # # # Save model
    # utils.save_model(model=model,
    #                  target_dir="../models",
    #                  model_name="Pretrained_model_efficientnet_b0.pth")

    # # Load model
    model.load_state_dict(torch.load(f="../models/Pretrained_model_efficientnet_b0.pth")) # noqa 5501

    # # # Plot loss curves
    # plot_loss_curves(result)

    # # Predict on images from test set
    img_to_plot = 10
    test_image_path_list = list(test_dir.glob("*/*.jpg"))
    img_path_sample = random.sample(population=test_image_path_list,
                                    k=img_to_plot)
    for img_path in img_path_sample:
        pred_and_plot_image(model=model,
                            image_path=img_path,
                            class_names=class_names)

    custom_image_path = "data/04-pizza-dad.jpeg"
    pred_and_plot_image(model=model,
                        image_path=custom_image_path,
                        class_names=class_names)

    # # # Make a confusion matrix - not working
    # test_data = datasets.ImageFolder(test_dir, transform=auto_transform)
    # print(test_data.classes)
    # x,y = make_and_plot_confusion_matrix(model=model,
    #                                dataloader=test_dataloader,
    #                                test_data=test_data,
    #                                class_names=class_names)
    # print(x.item(),y)


if __name__ == "__main__":
    main()
