import sys
import logging

import torch

from utils.save_model import save_model


class Trainer:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    @staticmethod
    def train_eval(model, criterion, optimizer, loader_train_set, loader_eval_set, epochs):
        Trainer.logger.info("Training started...")

        train_loss_averages = []
        train_accuracies = []

        eval_loss_averages = []
        eval_accuracies = []

        best_eval_loss = float('inf')
        for epoch in range(epochs):
            train_loss_average, train_accuracy = \
                Trainer.train(model, criterion, optimizer, loader_train_set, train_loss_averages, train_accuracies)

            eval_loss_average, eval_accuracy = \
                Trainer.eval(model, criterion, loader_eval_set, eval_loss_averages, eval_accuracies)

            best_eval_loss = Trainer.save_best_model(model, optimizer, epoch, eval_loss_average, best_eval_loss)

            Trainer.print_metrics(epochs, epoch, train_accuracy, train_loss_average, eval_accuracy, eval_loss_average)

        return train_loss_averages, train_accuracies, eval_loss_averages, eval_accuracies

    @staticmethod
    def save_best_model(model, optimizer, epoch, eval_loss_average, best_eval_loss):
        if eval_loss_average < best_eval_loss:
            best_eval_loss = eval_loss_average
            save_model(model, optimizer, epoch, best_eval_loss)
            Trainer.logger.info(f"Model saved at epoch {epoch + 1} | Best eval Loss: {best_eval_loss:.8f}")

        return best_eval_loss

    @staticmethod
    def train(model, criterion, optimizer, loader_set, loss_averages, accuracies):
        model.train()

        total_correct = 0
        total_loss = 0
        total_samples = 0

        for batch in loader_set:
            labels, loss, outputs = \
                Trainer.train_step(model, criterion, optimizer, batch)

            total_correct, total_loss, total_samples = \
                Trainer.gather_metrics(labels, loss, outputs, total_correct, total_loss, total_samples)

        train_loss_average, train_accuracy = \
            Trainer.compute_metrics(total_correct, total_loss, total_samples, loss_averages, accuracies)

        return train_loss_average, train_accuracy

    @staticmethod
    def train_step(model, criterion, optimizer, batch):
        inputs, labels = batch

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return labels, loss, outputs

    @staticmethod
    def eval(model, criterion, loader_set, loss_averages, accuracies):
        model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader_set:
                labels, loss, outputs = \
                    Trainer.eval_step(model, criterion, batch)

                total_correct, total_loss, total_samples = \
                    Trainer.gather_metrics(labels, loss, outputs, total_correct, total_loss, total_samples)

        loss_average, accuracy = \
            Trainer.compute_metrics(total_correct, total_loss, total_samples, loss_averages, accuracies)

        return loss_average, accuracy

    @staticmethod
    def eval_step(model, criterion, batch):
        inputs, labels = batch

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        return labels, loss, outputs

    @staticmethod
    def gather_metrics(labels, loss, outputs, total_correct, total_loss, total_samples):
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        return total_correct, total_loss, total_samples

    @staticmethod
    def compute_metrics(total_correct, total_loss, total_samples, loss_averages, accuracies):
        loss_average = total_loss / total_samples
        loss_averages.append(loss_average)

        accuracy = total_correct / total_samples
        accuracies.append(accuracy)

        return loss_average, accuracy

    @staticmethod
    def print_metrics(epochs, epoch, train_accuracy, train_loss_average, eval_accuracy, eval_loss_average):
        Trainer.logger.info(
            f"Epoch {epoch + 1}/{epochs}"
            f" | Training Loss Average: {train_loss_average:.7f}, Training Accuracy: {train_accuracy:.5f}"
            f" | Evaluation Loss Average: {eval_loss_average:.7f}, Evaluation Accuracy: {eval_accuracy:.5f}"
        )
