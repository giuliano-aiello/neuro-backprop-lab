import json
import logging
import sys
import torch.optim as optim
import torch.nn as nn

from utils.loader_dataset.loader_dataset import MNISTLoaderDataset
from trainer.irpropplus.irpropplus import IRpropPlus
from trainer.rpropminus.rpropminus import RpropMinus
from trainer.rpropplus.rpropplus import RpropPlus


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def load_config_training(model):
    with open("config/train/config.json", "r") as f:
        config = json.load(f)

    criterion            = config.get('criterion', 'cross_entropy')
    optimizer            = config.get('optimizer', 'rproppytorch')
    learning_rate        = config.get('learning_rate', 0.001)
    epochs               = config.get('epochs', 20)
    train_set_size       = config.get('train_set_size', 10000)
    train_batch_size     = config.get('train_batch_size', 2000)
    eval_set_size        = config.get('eval_set_size', 2500)
    eval_batch_size      = config.get('eval_batch_size', 500)

    MAX_MNIST_TRAINING_ITEMS = 60000
    if train_set_size + eval_set_size > MAX_MNIST_TRAINING_ITEMS:
        raise TypeError("Too many training items")

    if optimizer == 'rproppluspytorch':
        optimizer = optim.Rprop(model.parameters(), lr=learning_rate)
    elif optimizer == 'irpropplus':
        optimizer = IRpropPlus(model.parameters(), lr=learning_rate)
    elif optimizer == 'rpropminus':
        optimizer = RpropMinus(model.parameters(), lr=learning_rate)
    elif optimizer == 'rpropplus':
        optimizer = RpropPlus(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer not supported.")

    if criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Criterion not supported.")

    loader_train_set, loader_eval_set = \
        MNISTLoaderDataset.get_loader_dataset_train_eval(train_set_size, train_batch_size, eval_set_size, eval_batch_size)

    logger.info(f"Hyperparameters set.\tcriterion = {type(criterion).__name__}, optimizer = {get_optimizer_name(optimizer.__class__.__name__)}, learning rate = {learning_rate}, epochs = {epochs}, train set size = {train_set_size}, train batch size = {train_batch_size}, eval set size = {eval_set_size}, eval batch size = {eval_batch_size}")

    return criterion, optimizer, epochs, loader_train_set, loader_eval_set


def load_config_testing(optimizer):
    with open("config/test/config.json", "r") as f:
        config = json.load(f)

    criterion = config.get('criterion', 'cross_entropy')
    test_set_size       = config.get('test_set_size', 10000)
    test_batch_size     = config.get('test_batch_size', 1000)

    if criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Criterion not supported.")

    loader_test_set = MNISTLoaderDataset.get_loader_dataset_test(test_batch_size)

    log_model_info(criterion, optimizer, test_batch_size, test_set_size)

    return criterion, loader_test_set


def log_model_info(criterion, optimizer, test_batch_size, test_set_size):
    for param_group in optimizer.param_groups:
        lr = param_group.get('lr')

        etas = param_group.get('etas')
        etastring = etas
        if etastring is None:
            etaminus = param_group.get('etaminus')
            etaplus = param_group.get('etaplus')
            etastring = f"({etaminus}, {etaplus})"

    logger.info(f"criterion = {type(criterion).__name__}, "
                f"optimizer = {get_optimizer_name(optimizer.__class__.__name__)}, "
                f"learning rate = {lr}, "
                f"etas = {etastring}, "
                f" test set size = {test_set_size}, test batch size = {test_batch_size}")


def get_optimizer_name(optimizer_name):
    if optimizer_name == 'Rprop':
        optimizer_name = 'RpropPlusPyTorch'
    return optimizer_name
