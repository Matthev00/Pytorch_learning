import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt
from pathlib import Path


def count_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    total_time = end - start
    return (f"Trian time on {device}: {total_time:.3f} seconds")


class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
            )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)


class FashionMNISTModelV2(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.cllassifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.cllassifier(x)
        return x


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)
        label_pred = model(image)

        loss = loss_fn(label_pred, label)
        train_loss += loss
        train_acc += acc_fn(y_true=label,
                            y_pred=label_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return {"Train loss": f'{train_loss:.2f}',
            "Train acc": f'{train_acc:.2f}%'}


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              acc_fn,
              device: torch.device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for image, label in tqdm(dataloader):
            image, label = image.to(device), label.to(device)

            label_pred = model(image)

            loss += loss_fn(label_pred, label)
            acc += acc_fn(y_true=label,
                          y_pred=label_pred.argmax(dim=1))

        loss /= len(dataloader)
        acc /= len(dataloader)
    return {"Test_loss": f"{loss.item():.2f}",
            "Test_acc": f"{acc:.2f}%"
            }


def eval_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn,
               device: torch.device):
    """Returns a dict containing the results of model predicting on data loader.# noqa 5501
    """
    loss = 0
    acc = 0
    model.eval()
    with torch.inference_mode():
        for image, label in tqdm(dataloader):
            image, label = image.to(device), label.to(device)
            label_pred = model(image)

            loss += loss_fn(label_pred, label)
            acc += acc_fn(y_true=label,
                          y_pred=label_pred.argmax(dim=1))

        loss /= len(dataloader)
        acc /= len(dataloader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}


def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob.to("cpu"))
    return torch.stack(pred_probs)


def make_predictions_dataloader(model: torch.nn.Module,
                                data: DataLoader,
                                device: torch.device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for image, label in tqdm(data, desc="Makingpredictions..."):
            image, label = image.to(device), label.to(device)

            pred_logit = model(image)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0).argmax(dim=1)# noqa 5501

            pred_probs.append(pred_prob.to("cpu"))
    return torch.cat(pred_probs)


def plot_results(test_samples,
                 pred_classes,
                 test_labels,
                 class_names):
    torch.manual_seed(42)
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 3, 3
    for i, sample in enumerate(test_samples):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(sample.squeeze(), cmap="gray")

        pred_label = class_names[pred_classes[i]]
        true_label = class_names[test_labels[i]]
        title_text = f'Pred: {pred_label} | Truth: {true_label}'
        if pred_label == true_label:
            plt.title(title_text, fontsize=10, c="g")
        else:
            plt.title(title_text, fontsize=10, c="r")
        plt.axis(False)
    plt.show()


# Get data
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
    )

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
    )


class_names = train_data.classes


# Set hyperparameters
BATCH_SIZE = 32

# Turn data into iterables
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)


# Built Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# old model
torch.manual_seed(42)
model_0 = FashionMNISTModelV0(input_shape=28*28,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
# Mid model
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=28*28,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)

# New model
torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)


# Setup loss fn and optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_2.parameters(),
                            lr=0.1)

torch.manual_seed(42)

# train_time_on_gpu_start = timer()
# epochs = 3

# for epoch in tqdm(range(epochs)):
#     print(epoch)
#     train_step(model=model_2,
#                dataloader=train_dataloader,
#                loss_fn=loss_fn,
#                acc_fn=accuracy_fn,
#                optimizer=optimizer,
#                device=device)

#     test_step(model=model_2,
#               dataloader=train_dataloader,
#               loss_fn=loss_fn,
#               acc_fn=accuracy_fn,
#               device=device)

# train_time_on_gpu_end = timer()
# train_time_gpu_model_2 = count_train_time(train_time_on_gpu_start,
#                                           train_time_on_gpu_end,
#                                           device)
# print(train_time_gpu_model_2)
# print(eval_model(model=model_2,
#                  dataloader=test_dataloader,
#                  loss_fn=loss_fn,
#                  acc_fn=accuracy_fn,
#                  device=device))


# Save model
model_path = Path('models')
model_name = '03_pytorch_computer_vision.pth'
model_save_path = model_path / model_name

# torch.save(obj=model_2.state_dict(), f=model_save_path)

# Load model
model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
model_2.load_state_dict(torch.load(f=model_save_path))

print(eval_model(model=model_2,
                 dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 acc_fn=accuracy_fn,
                 device=device))


# Test on random samples
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model=model_2,
                              data=test_samples,
                              device=device)
pred_classes = pred_probs.argmax(dim=1)

# Plot test
plot_results(test_samples=test_samples,
             pred_classes=pred_classes,
             test_labels=test_labels,
             class_names=class_names)

y_preds = make_predictions_dataloader(model=model_2,
                                      data=test_dataloader,
                                      device=device)


# Set up confusion instance and compere oredictions to targets
conf_mat = ConfusionMatrix(num_classes=len(class_names),
                           task="multiclass")
conf_mat_tensor = conf_mat(preds=y_preds,
                           target=test_data.targets)

fig, ax = plot_confusion_matrix(conf_mat=conf_mat_tensor.numpy(),
                                class_names=class_names,
                                figsize=(10, 7))
plt.show()
