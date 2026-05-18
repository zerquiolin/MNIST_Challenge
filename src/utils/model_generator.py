# Modules
import os
import sys

import torch

# Mnist Dataloaders
from src.data.mnist import load_mnist_dataloaders

# Custom Neural Network
from src.models.mnist_cnn import MNISTCNN

# Training Loop
from src.training.engine import train

# Utils
from src.utils.device import get_device

# Load MNIST data loaders
train_loader, eval_loader, test_loader, metadata = load_mnist_dataloaders()
# Instatiation of the model
model = MNISTCNN()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Loss Function
criterion = torch.nn.CrossEntropyLoss()
# Identify device
device = get_device()
# Train Model
model, history = train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    epochs=22,
    device=device,
)
# Output Folder
output_dir = "../artifacts/mnist_classifier"
# Make sure the data directory exists
os.makedirs(output_dir, exist_ok=True)
# Save model
torch.save(model.state_dict(), f"{output_dir}.pt")
