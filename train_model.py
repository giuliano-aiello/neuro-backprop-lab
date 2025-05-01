from utils.config_loader import load_config_training
from utils.plot_metrics import plot_metrics
from model.model import Model
from trainer.trainer import Trainer


model = Model()

criterion, optimizer, epochs, loader_train_set, loader_eval_set = \
    load_config_training(model)

training_losses, training_accuracies, evaluation_losses, evaluation_accuracies = \
    Trainer.train_eval(model, criterion, optimizer, loader_train_set, loader_eval_set, epochs)

plot_metrics(training_losses, training_accuracies, evaluation_losses, evaluation_accuracies)
