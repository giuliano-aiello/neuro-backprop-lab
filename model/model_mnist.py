import torch.nn as nn

class ModelMNIST(nn.Module):

    MNIST_IMAGE_SIZE = 28 * 28
    MNIST_NUM_CLASSES = 10

    def __init__(self):
        super(ModelMNIST, self).__init__()

        self.layer_input = nn.Flatten()
        self.layer_fully_connected = nn.Linear(ModelMNIST.MNIST_IMAGE_SIZE, 128)
        self.layer_output = nn.Linear(128, ModelMNIST.MNIST_NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(self.layer_fully_connected(x))
        x = self.layer_output(x)

        return x