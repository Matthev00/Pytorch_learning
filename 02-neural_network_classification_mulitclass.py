from torch import nn
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from torchmetrics import Accuracy
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary


class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        """_summary_

        Args:
            input_features (int): Number of input features
            output_features (int): Number of output features
            hidden_units (int): Number of hidden units between layers
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42


# Make data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

# Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# Split data into training and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,# noqa 5501
                                                                        random_state=RANDOM_SEED)# noqa 5501

# Visulaize
# plt.figure(figsize=(10, 7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

# Create device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Put data on the right device
X_blob_train = X_blob_train.to(device)
y_blob_train = y_blob_train.to(device)
X_blob_test = X_blob_test.to(device)
y_blob_test = y_blob_test.to(device)

# Create instance of a BlobModel and send it to the right device
model_4 = BlobModel(input_features=2,
                    output_features=4,
                    hidden_units=8).to(device)

# Set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)


# Training model
torch.manual_seed(42)

model_4.eval()
with torch.inference_mode():
    # Forward pass
    y_logits = model_4(X_blob_test)
    y_pred_probs = torch.softmax(y_logits, dim=1)
    y_preds = torch.argmax(y_pred_probs, dim=1)

# print(y_preds[:5])

epochs = 100

# Training model
torch.manual_seed(42)

# Build training loop
for epoch in range(epochs):
    # Training
    model_4.train()

    # Forward pass
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(input=y_logits, dim=1).argmax(dim=1)

    # Calculate loss/accuracy
    loss = loss_fn(y_logits, y_blob_train)

    # Optimizer zero grad
    optimizer.zero_grad()

    # Loss backwards
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Testing
    model_4.eval()
    with torch.inference_mode():
        # Forward pass
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(input=test_logits, dim=1).argmax(dim=1)

        # Calculate test loss
        test_loss = loss_fn(test_logits, y_blob_test)

    # Print what's happening
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.5f} |v Test loss: {test_loss:.5f}')# noqa 5501


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()


torch_metric_accuracy = Accuracy(task="multiclass", num_classes=4).to(device)
print(torch_metric_accuracy(test_pred, y_blob_test))
