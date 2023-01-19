import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(filename="training_log.log")
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_MNIST_data():
    # Download training data from open datasets.
    training_data_ = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data_ = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data_, test_data_

def get_data_shape(test_dataloader_):
    for X, y in test_dataloader_:
        print(f"Shape of X [Batch Size, Channel, Height, Width]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

# Configure hyper-parameters
class Config:
    def __init__(self, loss_function_, optimizer_, epoch_, batch_size_=64, ):
        self.device_ = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.batch_size_ = batch_size_
        self.loss_function_ = loss_function_
        self.optimizer_ = optimizer_
        self.epoch_ = epoch_

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model_, loss_fn_, optimizer_):
    size = len(dataloader.dataset)
    model_.train()

    time_start = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model_(X)
        loss = loss_fn_(pred, y)

        # Backpropagation
        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()

        if batch % 100 == 0:
            time_gap = time.time() - time_start
            time_start = time.time()
            loss, current = loss.item(), batch * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] [{time_gap:>2f}]")

def test(dataloader, model_, loss_fn_):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model_.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model_(X)
            test_loss += loss_fn_(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    my_conf = Config(
        batch_size_=64,
        loss_function_=nn.CrossEntropyLoss(),
        optimizer_=torch.optim.SGD,
        epoch_=5
    )
    # Data pre-procession
    training_data, test_data = get_MNIST_data()
    batch_size = my_conf.batch_size_

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Get cpu or gpu device for training.
    device = my_conf.device_
    logger.info(f"Using {device} device")

    model = NeuralNetwork().to(device)
    logger.info(model)

    loss_fn = my_conf.loss_function_
    optimizer = my_conf.optimizer_(model.parameters(), lr=1e-3)

    epochs = my_conf.epoch_
    start = time.time()
    for t in range(epochs):
        gap = time.time() - start
        start = time.time()
        logger.info(f"Epoch {t + 1}\n-------------------------------{gap:>3f}")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    logger.info("Done!")