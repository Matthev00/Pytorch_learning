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


def main():
    args = utils.parse_arguments()
    BATCH_SIZE = args.batch_size
    utils.set_seeds(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # # # Download pizza, steak, sushi images from GitHub
    data_20_percent_path = download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip", # noqa 5501
        destination="pizza_steak_sushi_20_percent")

    # # Set data paths
    # 09_Model_Deployment\data\pizza_steak_sushi_20_percent
    # data_20_precent_path = Path("/data/pizza_steak_sushi_20_percent")
    train_dir = data_20_percent_path / "train"
    test_dir = data_20_percent_path / "test"

    # # Create a model
    effnetb2, effnetb2_transforms = model_builder.create_effnetb2(
        out_features=3,
        device=device)
    # print(summary(model=effnetb2,
    #               input_size=(1, 3, 224, 224),
    #               col_names=["input_size", "output_size", "num_params", "trainable"], # noqa 5501
    #               col_width=15,
    #               row_settings=["var_names"]))

    # # Setup Dataloaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders( # noqa 5501
        train_dir=train_dir,
        test_dir=test_dir,
        transform=effnetb2_transforms,
        batch_size=BATCH_SIZE)

    # # Train or load pretrained model
    model_path = "../models/07_effnetb2_data_20_precent_10_epochs.pth"
    effnetb2.load_state_dict(torch.load(model_path))
    model_size = Path(model_path).stat().st_size // (1024*1024) # noqa 5501

    # # Get test data paths
    test_data_paths = list(Path(test_dir).glob(pattern="*/*.jpg"))

    # # Make predictions on test data
    test_pred_dict = utils.pred_and_store(
        paths=test_data_paths,
        model=effnetb2,
        transform=effnetb2_transforms,
        class_names=class_names)

    # # Print results
    # effnetb2_test_pred_df = pd.DataFrame(test_pred_dict)
    # print(effnetb2_test_pred_df.head(5))
    # print(effnetb2_test_pred_df.correct.value_counts())
    # avg_time = effnetb2_test_pred_df.prediction_time.mean()
    # print(f"Avrerage time: {avg_time}")

    # # Test on random img
    random_img_path = random.sample(population=test_data_paths,
                                    k=1)[0]
    img = Image.open(fp=random_img_path)

    def predict(img: Image):

        start_time = timer()

        transformed_img = effnetb2_transforms(img).unsqueeze(0).to(device)

        effnetb2.to(device)
        effnetb2.eval()

        with torch.inference_mode():
            pred_logit = effnetb2(transformed_img)
            pred_prob = torch.softmax(input=pred_logit, dim=1)

        pred_labels_and_probs = {class_names[i]: float(pred_prob[0][i]) for i in range(len(class_names))} # noqa 5501

        pred_time = round(timer() - start_time, 5)

        return pred_labels_and_probs, pred_time

    # pred_dict, pred_time = predict(img=img)
    # print(pred_dict)
    # print(pred_time)

    example_list = [[str(path)] for path in random.sample(
        population=test_data_paths,
        k=3)]

    # print(example_list)

    title = "FoodVision Mini 🍕🥩🍣"
    description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi." # noqa 5501
    article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)." # noqa 5501

    # # Create a Gradio
    demo = gr.Interface(fn=predict,  # mapping function from input to output
                        inputs=gr.Image(type="pil"),  # what are the inputs?
                        outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs? # noqa 5501
                                 gr.Number(label="Prediction time (s)")],  # our fn has two outputs, therefore we have two outputs # noqa 5501
                        examples=example_list,
                        title=title,
                        description=description,
                        article=article)

    demo.launch(debug=False,
                share=True)


if __name__ == "__main__":
    main()
