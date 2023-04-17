from torch import nn
import torch
from matplotlib import pyplot as plt
from pathlib import Path
import requests
from sklearn.datasets import make_circles
import pandas as pd
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary


# Construct a model that subclass Model.nn
class CircleModelV0(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super().__init__()
        # Create 2 nn.Linear layers capable of handling the shape of pur data
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    # Define a forward() method that outlines the forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x))


# Create better module fo this task
class CircleModelV1(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super().__init__()
        # Create 2 nn.Linear layers capable of handling the shape of pur data
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    # Define a forward() method that outlines the forward pass
    def forward(self, x):
        z = self.layer_1(x)
        z = self.layer_2(z)
        z = self.layer_3(z)
        return z


# Calculate accuracy - out of 100 examples, what percantage does our model get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) / 100
    return acc


# Make data
n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)


# Visualize
circles = pd.DataFrame({"x1": X[:, 0],
                        'x2': X[:, 1],
                       'label': y})
# print(circles.head(10))

# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             c=y,
#             cmap=plt.cm.RdYlBu)
# plt.show()

# print(X.shape)
# X_sample = X[0]
# y_sample = y[0]
# print(X_sample, y_sample)
# print(X_sample.shape, y_sample.shape)


# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
# print(X[:5], y[:5])


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


# Building a Model
# Create device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)

# Create an instance of Model class and put it in the right device
# model_0 = CircleModelV0().to(device)

# Create a model0 in simplier way using nn.Sequential()
model0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)
# print(model0.state_dict())


# Set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model0.parameters(),
                            lr=0.1)


# Training model

# View first 5 outputs of the forward passon the test data
model0.eval()
with torch.inference_mode():
    y_logits = model0(X_test.to(device))[:5]
# print(y_logits)
# print(y_test[:5])

# Use sigmoid activation function on our model logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)

# Find the predicted labels
y_preds = torch.round(y_pred_probs)

# Get rid of extra dimentions
y_preds = y_preds.squeeze()
# print(y_preds)


# Build training loop and testing loop
torch.manual_seed(42)

epochs = 100

# Put data to the taret device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training loop
for epoch in range(epochs):
    # Training
    model0.train()

    # Forward pass
    y_logits = model0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backwards
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Testing
    model0.eval()
    with torch.inference_mode():
        # Forward pass
        test_logits = model0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(y_logits))

        # Calculate test loss
        test_loss = loss_fn(test_logits, y_test)

    # Print what's happening
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.5f} |v Test loss: {test_loss:.5f}')


# Visiualize Visiualize Visiualize
# Download helper fn from repo
if not Path("helper_functions.py").is_file():
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as file:
        file.write(request.content)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model0, X_test, y_test)
plt.show()

model_1 = CircleModelV1().to(device)
print(model_1)