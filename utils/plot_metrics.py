import matplotlib.pyplot as plt


def plot_metrics(training_losses, training_accuracies, evaluation_losses, evaluation_accuracies):
    loss(training_losses, stage="Training")
    accuracy(training_accuracies, stage="Training")

    loss(evaluation_losses, stage="Evaluation")
    accuracy(evaluation_accuracies, stage="Evaluation")

def loss(losses, stage):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{stage.capitalize()}')
    plt.tight_layout()
    plt.show()

def accuracy(accuracies, stage):
    plt.plot(accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{stage.capitalize()}')
    plt.tight_layout()
    plt.show()