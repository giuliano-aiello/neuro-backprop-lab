import torch
import logging
import sys


class Tester:

    @staticmethod
    def test(model, criterion, loader_set):
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        logger = logging.getLogger(__name__)
        logger.info("Testing started...")

        model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader_set:
                labels, loss, outputs = Tester.eval_step(model, criterion, batch)
                total_correct, total_loss, total_samples = \
                    Tester.gather_metrics(labels, loss, outputs, total_correct, total_loss, total_samples)

        loss_average, accuracy = \
            Tester.compute_metrics(total_correct, total_loss, total_samples)

        Tester.print_metrics(loss_average, accuracy)

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
    def compute_metrics(total_correct, total_loss, total_samples):
        loss_averages = []
        loss_average = total_loss / total_samples
        loss_averages.append(loss_average)

        accuracies = []
        accuracy = total_correct / total_samples
        accuracies.append(accuracy)

        return loss_average, accuracy

    @staticmethod
    def print_metrics(test_loss_average, test_accuracy):
        print(f"Test Loss Average: {test_loss_average:.4f}, Test Accuracy: {test_accuracy:.4f}")
