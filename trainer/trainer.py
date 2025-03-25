import torch


class Trainer:

    @staticmethod
    def train_eval(model, loader_train_set, loader_eval_set, criterion, optimizer, epochs):
        train_loss_averages = []
        train_accuracies = []

        eval_loss_averages = []
        eval_accuracies = []

        for epoch in range(epochs):
            train_loss_average, train_accuracy = \
                Trainer.train(model, criterion, optimizer, loader_train_set, train_loss_averages, train_accuracies)

            eval_loss_average, eval_accuracy = \
                Trainer.eval(criterion, eval_accuracies, eval_loss_averages, loader_eval_set, model)

            Trainer.print_metrics(epochs, epoch, train_accuracy, train_loss_average, eval_accuracy, eval_loss_average)

        return train_loss_averages, train_accuracies, eval_loss_averages, eval_accuracies

    @staticmethod
    def train(model, criterion, optimizer, loader_train_set, train_loss_averges, train_accuracies):
        model.train()

        total_correct = 0
        total_loss = 0
        total_samples = 0

        for batch in loader_train_set:
            labels, loss, outputs = \
                Trainer.train_step(model, batch, criterion, optimizer)

            total_correct, total_loss, total_samples = \
                Trainer.gather_metrics(labels, loss, outputs, total_correct, total_loss, total_samples)

        train_loss_average, train_accuracy = \
            Trainer.compute_metrics(total_correct, total_loss, total_samples, train_loss_averges, train_accuracies)

        return train_loss_average, train_accuracy

    @staticmethod
    def train_step(model, batch, criterion, optimizer):
        inputs, labels = batch

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return labels, loss, outputs

    @staticmethod
    def eval(criterion, accuracies, loss_averages, loader_eval_set, model):
        model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader_eval_set:
                labels, loss, outputs = \
                    Trainer.eval_step(batch, criterion, model)

                total_correct, total_loss, total_samples = \
                    Trainer.gather_metrics(labels, loss, outputs, total_correct, total_loss, total_samples)

        loss_average, accuracy = \
            Trainer.compute_metrics(total_correct, total_loss, total_samples, loss_averages, accuracies)

        return loss_average, accuracy

    @staticmethod
    def eval_step(batch, criterion, model):
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
    def compute_metrics(total_correct, total_loss, total_samples, loss_avgerages, accuracies):
        loss_average = total_loss / total_samples
        loss_avgerages.append(loss_average)

        accuracy = total_correct / total_samples
        accuracies.append(accuracy)

        return loss_average, accuracy

    @staticmethod
    def print_metrics(epochs, epoch, train_accuracy, train_loss_average, eval_accuracy, eval_loss_average):
        print(
            f"Epoch {epoch + 1}/{epochs}"
            f" | Training Loss Average: {train_loss_average:.4f}, Training Accuracy: {train_accuracy:.4f}"
            f" | Evaluation Loss Average: {eval_loss_average:.4f}, Evaluation Accuracy: {eval_accuracy:.4f}"
        )