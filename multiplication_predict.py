

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

def dataset_generator():
    #If you want a different number other than 2 you can do so
    train_set = {}
    for i in range(1001):
        train_set[(i, 2)] = i * 2
    return train_set

ds = dataset_generator()

ds_features = list(ds.keys())
ds_target = list(ds.values())

print(ds_features[:5])
print(ds_target[:5])

train_split = int(0.8 * len(ds)) # 80% of data used for training set, 20% for testing

x_train , y_train = ds_features[:train_split],ds_target[:train_split]
x_test, y_test = ds_features[train_split:] , ds_target[train_split:]

"""# Convert the dataset to tensors"""

import random

train_ds = dict(list(ds.items())[:train_split])
test_ds = dict(list(ds.items())[train_split:])
shuffle_list = list(ds.items())
random.shuffle(shuffle_list)
train_ds = dict(shuffle_list)
print(train_ds)

trainds_tensor = {torch.tensor(key,dtype=torch.float32):torch.tensor(value,dtype=torch.float32)  for key, value in train_ds.items()}
testds_tensor = {torch.tensor(key,dtype=torch.float32):torch.tensor(value,dtype=torch.float32) for key, value in test_ds.items()}

trainds_tensor

len(trainds_tensor), len(testds_tensor)

criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Adam optimizer

"""# Note to self check out tensordict (https://github.com/pytorch-labs/tensordict) in case normal dict didnt work"""

num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    for key, value in trainds_tensor.items():
        optimizer.zero_grad()

        key = key.to(device)  # Move input data to device
        value = value.to(device)  # Move target data to device

        predicted = model(key)
        #print(f"predicted:{predicted} for {key}")
        loss = criterion(predicted, value)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(trainds_tensor)
    print(f'Epoch [{epoch+1}/{num_epochs}], Avg. Training Loss: {avg_loss:.4f}')

print('Training finished!')

model.eval()  # Set the model to evaluation mode
total_test_loss = 0.0

with torch.no_grad():
    for key, value in testds_tensor.items():
        key = key.to(device)  # Move input data to device
        value = value.to(device)  # Move target data to device
        predicted = model(key)
        loss = criterion(predicted, value)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(testds_tensor)
print(f'Avg. Testing Loss: {avg_test_loss:.4f}')

print('Testing finished!')

# Sample input
x = [100,2]

x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

res = model(x_tensor)

print(res)