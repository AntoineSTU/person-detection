import logging

from .metrics.metrics import accuracy, f_one_score
from .graphs.graphs import roc_curve_plot, confusion_matrix_plot, plot_train_val_loss


def log_metrics(true_labels, predictions):
    """
    This function's goal is to compute some metrics to assess the model.
    @type true_labels: Numpy array
    @param true_labels: The real labels of the images
    @type predictions: Numpy array
    @param predictions: The labels predicted by the model (same order as the true labels)

    @rtype: {"accuracy": float, "f1_score": float}
    @return: All the labels
    """

    acc = accuracy(true_labels, predictions)
    print("Accuracy value: {}".format(acc))
    logging.info("Accuracy value: {}".format(acc))

    f1_score = f_one_score(true_labels, predictions)
    print("f1 score: {}".format(f1_score))
    logging.info("f1 score: {}".format(f1_score))

    return {"accuracy": acc, "f1_score": f1_score}


def plot_graphs(true_labels, predictions, figures_save_path):
    """
    This function's goal is to draw some graphs to assess the model.
    @type true_labels: Numpy array
    @param true_labels: The real labels of the images
    @type predictions: Numpy array
    @param predictions: The labels predicted by the model (same order as the true labels)
    @type figures_save_path: string
    @param figures_save_path: Path to the folder to save the figures

    @rtype: None
    @return: Nothing (the figures are saved in some files)
    """

    roc_curve_plot(true_labels, predictions, figures_save_path)
    confusion_matrix_plot(true_labels, predictions, figures_save_path)


def visualize(true_labels, predictions, history, figures_save_path):
    """
    Show some metrics and plot some graphs from predictions and true labels.
    @type true_labels: Numpy array
    @param true_labels: The real labels of the images
    @type predictions: Numpy array
    @param predictions: The labels predicted by the model (same order as the true labels)
    @type history: Keras training history
    @param history: The training history on the model
    @type figures_save_path: string
    @param figures_save_path: Path to the folder to save the figures

    @rtype: {"accuracy": float, "f1_score": float}
    @return: The accuracy and f1 score
    """

    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1

    metrics = log_metrics(true_labels, predictions)
    plot_graphs(true_labels, predictions, figures_save_path)
    plot_train_val_loss(history, figures_save_path)
    return metrics
