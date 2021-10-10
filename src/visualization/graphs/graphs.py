import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import numpy as np


def confusion_matrix_plot(true_labels, predictions, figures_save_path):
    """
    This function's goal is to draw the confusion matrix.
    @type true_labels: Numpy array
    @param true_labels: The real labels of the images
    @type predictions: Numpy array
    @param predictions: The labels predicted by the model (same order as the true labels)
    @type figures_save_path: string
    @param figures_save_path: Path to the folder to save the figures

    @rtype: None
    @return: Nothing (the figure is saved in a file)
    """

    cf_matrix = confusion_matrix(true_labels, predictions)
    group_names = ["True Negative", "False Positive",
                   "False Negative", "True Positive"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    fig = plt.figure()
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues',
                xticklabels=False, yticklabels=False)
    plt.title("confusion matrix")
    plt.savefig("{}/confusion_matrix.png".format(figures_save_path))


def roc_curve_plot(true_labels, predictions, figures_save_path):
    """
    This function's goal is to draw the ROC curve.
    @type true_labels: Numpy array
    @param true_labels: The real labels of the images
    @type predictions: Numpy array
    @param predictions: The labels predicted by the model (same order as the true labels)
    @type figures_save_path: string
    @param figures_save_path: Path to the folder to save the figures

    @rtype: None
    @return: Nothing (the figure is saved in a file)
    """

    fpr, tpr, _ = roc_curve(true_labels, predictions)
    fig = plt.figure()
    plt.plot(fpr, tpr, 'orange')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("ROC curve")
    plt.savefig("{}/ROC.png".format(figures_save_path))


def plot_train_val_loss(history, figures_save_path):
    """
    This function's goal is to plot the train and validation loss along the training step.
    @type history: Keras training history
    @param history: The training history on the model
    @type figures_save_path: string
    @param figures_save_path: Path to the folder to save the figures

    @rtype: None
    @return: Nothing (the figure is saved in a file)
    """

    fig = plt.figure()
    plt.plot(history.history["loss"], label="trainning loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.legend()
    plt.title("Train and Validation Loss")
    plt.savefig("{}/train_val_loss.png".format(figures_save_path))

    fig = plt.figure()
    plt.plot(history.history["acc"], label="training accuracy")
    plt.plot(history.history["val_acc"], label="validation accuracy")
    plt.legend()
    plt.title("Train and Validation Accuracy")
    plt.savefig("{}/train_val_acc.png".format(figures_save_path))
