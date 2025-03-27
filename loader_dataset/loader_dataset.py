import torch
import torchvision
import torchvision.transforms as transforms


class LoaderDataset(torch.utils.data.Dataset):

    @staticmethod
    def get_mnist_loader_dataset(train_mode=True, batch_size=256, subset_size=None):
        dataset = LoaderDataset.__get_dataset(train_mode, subset_size)

        loader_dataset = LoaderDataset.__get_loader_dataset(dataset, train_mode, batch_size)

        return loader_dataset

    @staticmethod
    def __get_loader_dataset(dataset, train_mode, batch_size):
        shuffle = train_mode

        loader_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return loader_dataset

    @staticmethod
    def __get_dataset(train_mode, subset_size: None):
        transform = LoaderDataset.__get_transform()

        dataset = torchvision.datasets.MNIST(root='./data', train=train_mode, download=True, transform=transform)

        if subset_size is not None:
            dataset = torch.utils.data.Subset(dataset, range(subset_size))

        return dataset

    @staticmethod
    def __get_transform():
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        return transform
