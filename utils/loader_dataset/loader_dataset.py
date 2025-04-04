import torch
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms


class MNISTLoaderDataset(torch.utils.data.Dataset):

    @staticmethod
    def get_loader_dataset_test(batch_size=256):
        dataset_test = MNISTLoaderDataset.__get_dataset(train_mode=False)

        loader_dataset_test = MNISTLoaderDataset.__get_loader_dataset(dataset_test, batch_size=batch_size, shuffle=False)

        return loader_dataset_test

    @staticmethod
    def get_loader_dataset_train_eval(dataset_size_train=10000, batch_size_train=2000, dataset_size_eval=2500, batch_size_eval=500):
        training_set = MNISTLoaderDataset.__get_dataset()

        dataset_train, dataset_eval = random_split(training_set, [dataset_size_train, dataset_size_eval])

        loader_dataset_train = MNISTLoaderDataset.__get_loader_dataset(dataset_train, batch_size=batch_size_train, shuffle=True)
        loader_dataset_eval = MNISTLoaderDataset.__get_loader_dataset(dataset_eval, batch_size=batch_size_eval, shuffle=False)

        return loader_dataset_train, loader_dataset_eval

    @staticmethod
    def __get_loader_dataset(dataset, batch_size=256, shuffle=True):
        loader_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return loader_dataset

    @staticmethod
    def __get_dataset(train_mode=True):
        dataset = torchvision.datasets.MNIST(root='./data', train=train_mode, download=True, transform=MNISTLoaderDataset.__get_transform())

        return dataset

    @staticmethod
    def __get_transform():
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        return transform
