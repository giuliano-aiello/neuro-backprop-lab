import matplotlib.pyplot as plt


def plot_metrics(training_losses, training_accuracies, evaluation_losses, evaluation_accuracies):
    loss(training_losses)
    accuracy(training_accuracies)

    loss(evaluation_losses)
    accuracy(evaluation_accuracies)

def loss(losses):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.title('Network Enhancement (RProp) on MNIST database')
    plt.show()

def accuracy(accuracies):
    plt.plot(accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.title('Network Enhancement (RProp) on MNIST database')
    plt.show()