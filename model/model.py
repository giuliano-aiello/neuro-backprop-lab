import torch.nn as nn

class Model(nn.Module):

    MNIST_IMAGE_SIZE = 28 * 28
    MNIST_NUM_CLASSES = 10

    def __init__(self):
        super(Model, self).__init__()
        self.layer_input = nn.Flatten()
        self.layer_fully_connected = nn.Linear(Model.MNIST_IMAGE_SIZE, 128)
        self.layer_output = nn.Linear(128, Model.MNIST_NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(self.layer_fully_connected(x))
        x = self.layer_output(x)
        return x