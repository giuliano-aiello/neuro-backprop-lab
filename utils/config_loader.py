import json
import logging
import sys
import torch.optim as optim
import torch.nn as nn

from trainer.irpropplus.irpropplus import IRpropPlus
from trainer.rpropplus.rpropplus import RpropPlus


def load_config_training(model):
    with open("config/train/config.json", "r") as f:
        config = json.load(f)

    criterion     = config.get('criterion', 'cross_entropy')
    optimizer     = config.get('optimizer', 'rprop')
    learning_rate = config.get('learning_rate', 0.001)
    epochs        = config.get('epochs', 20)

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

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    logger.info(f"Hyperparameters set.\tcriterion = {type(criterion).__name__}, optimizer = {type(optimizer).__name__}, learning rate = {learning_rate}, epochs = {epochs}")

    return criterion, optimizer, epochs

def load_config_testing():
    with open("config/test/config.json", "r") as f:
        config = json.load(f)

    criterion = config.get('criterion', 'cross_entropy')

    if criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Criterion not supported.")

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    logger.info(f"criterion = {type(criterion).__name__}")

    return criterion