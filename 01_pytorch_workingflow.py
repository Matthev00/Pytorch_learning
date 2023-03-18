from torch import nn
import torch
from matplotlib import pyplot as plt
from pathlib import Path


class LinearRegressionModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() to create model parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

        # Creating parameters on my own
        # self.weights = nn.Parameter(torch.rand(1,
        #                                        requires_grad=True,
        #                                        dtype=torch.float))
        # self.bias = nn.Parameter(torch.rand(1,
        #                                     requires_grad=True,
        #                                     dtype=torch.float))

    # Used with parameter created on my own
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.weights * x + self.bias

    # Used with parameter created by nn.Linear
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


def plotPredictions(trainData,
                    trainLabels,
                    testData,
                    testLabels,
                    predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(trainData, trainLabels, c='b', s=4, label="Training Data")
    plt.scatter(testData, testLabels, c='g', s=4, label="Testing Data")
    if predictions is not None:
        plt.scatter(testData, predictions, c='r', s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()


# Create device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)

# Create known parameters
weight = 0.7
bias = 0.3

# Create X and y
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Create a train/test split
trainSplit = int(0.8*len(X))
X_train, y_train = X[:trainSplit], y[:trainSplit]
X_test, y_test = X[trainSplit:], y[trainSplit:]

# Set the manual seed
torch.manual_seed(42)

# Put data on the right device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Creating model
model_0 = LinearRegressionModule().to(device)
with torch.inference_mode():
    y_preds = model_0(X_test)

# Set up a loss function
loss_fn = nn.L1Loss()

# Set optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

# Building training loop
epochs = 300
torch.manual_seed(42)
for epoch in range(epochs):
    # Set model to training mode
    model_0.train()

    # Forward pass
    y_preds = model_0(X_train)

    # Calculate loss
    loss = loss_fn(y_preds, y_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Perform backpropagation on the loss with respect to the params of model
    loss.backward()

    # Step opt
    optimizer.step()

    # Testing
    model_0.eval()

    with torch.inference_mode():
        test_preds = model_0(X_test)

        test_loss = loss_fn(test_preds, y_test)
        if epoch % 10 == 0:
            print(test_loss)

# Turn into evaluation mode
model_0.eval()

# Make predictions
with torch.inference_mode():
    y_preds = model_0(X_test)

# Making a plot
plotPredictions(X_train.cpu(), y_train.cpu(),
                X_test.cpu(), y_test.cpu(),
                predictions=y_preds.cpu())

# Saving model
# 1.Create dir
model_path = Path('models')
model_path.mkdir(parents=True, exist_ok=True)

# 2.Create model save path
model_name = '01_pytorch_workflow_model_0.pth'
model_save_path = model_path / model_name

# 3. Save state dict of model
torch.save(obj=model_0.state_dict(), f=model_save_path)

# Loading model
loaded_model_0 = LinearRegressionModule().to(device)
loaded_model_0.load_state_dict(torch.load(f=model_save_path))

# Evaluate loaded model
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_0_preds = loaded_model_0(X_test)
print(y_preds == loaded_model_0_preds)
