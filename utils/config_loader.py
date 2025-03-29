import json
import logging
import sys
import torch.optim as optim
import torch.nn as nn

from loader_dataset.loader_dataset import LoaderDataset
from trainer.irpropplus.irpropplus import IRpropPlus
from trainer.rpropplus.rpropplus import RpropPlus


def load_config_training(model):
    with open("config/train/config.json", "r") as f:
        config = json.load(f)

    criterion            = config.get('criterion', 'cross_entropy')
    optimizer            = config.get('optimizer', 'rprop')
    learning_rate        = config.get('learning_rate', 0.001)
    epochs               = config.get('epochs', 20)
    train_set_size       = config.get('train_set_size', 10000)
    train_batch_size     = config.get('train_batch_size', 2000)
    eval_set_size        = config.get('eval_set_size', 2500)
    eval_batch_size      = config.get('eval_batch_size', 500)

    if optimizer == 'rprop':
        optimizer = optim.Rprop(model.parameters(), lr=learning_rate)
    elif optimizer == 'rpropplus':
        optimizer = RpropPlus(model.parameters(), lr=learning_rate)
    elif optimizer == 'irpropplus':
        optimizer = IRpropPlus(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer not supported.")

    if criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Criterion not supported.")

    loader_train_set = LoaderDataset.get_mnist_loader_dataset(True, train_batch_size, train_set_size)
    loader_eval_set  = LoaderDataset.get_mnist_loader_dataset(False, eval_batch_size, eval_set_size)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    logger.info(f"Hyperparameters set.\tcriterion = {type(criterion).__name__}, optimizer = {type(optimizer).__name__}, learning rate = {learning_rate}, epochs = {epochs}, train set size = {train_set_size}, train batch size = {train_batch_size}, eval set size = {eval_set_size}, eval batch size = {eval_batch_size}")
    return criterion, optimizer, epochs, loader_train_set, loader_eval_set

def load_config_testing():
    with open("config/test/config.json", "r") as f:
        config = json.load(f)

    criterion = config.get('criterion', 'cross_entropy')
    test_set_size       = config.get('test_set_size', 10000)
    test_batch_size     = config.get('test_batch_size', 1000)

    if criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Criterion not supported.")

    loader_test_set = LoaderDataset.get_mnist_loader_dataset(False, test_batch_size, test_set_size)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    logger.info(f"criterion = {type(criterion).__name__}, test set size = {test_set_size}, test batch size = {test_batch_size}")

    return criterion, loader_test_set