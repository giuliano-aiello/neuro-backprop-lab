import json
import logging
import sys
import torch.optim as optim
import torch.nn as nn


def load_config_model(model):
    with open("config/config.json", "r") as f:
        config = json.load(f)

    criterion     = config.get('criterion', 'cross_entropy')
    optimizer     = config.get('optimizer', 'rprop')
    learning_rate = config.get('learning_rate', 0.001)
    epochs        = config.get('epochs', 20)

    if optimizer == 'rprop':
        optimizer = optim.Rprop(model.parameters(), lr=learning_rate)
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
