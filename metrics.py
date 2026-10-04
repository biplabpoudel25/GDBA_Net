import numpy as np


def calculate_metrics(pred, gt, epsilon=1e-7):
    """
    Calculate binary segmentation metrics for a single prediction.

    Parameters
    ----------
    pred : numpy.ndarray
        Binary predicted mask containing values 0 and 1.

    gt : numpy.ndarray
        Binary ground-truth mask containing values 0 and 1.

    epsilon : float, optional
        Small value used to avoid division by zero.

    Returns
    -------
    dict
        Dictionary containing Dice, IoU, Precision, Recall,
        and Specificity.
    """
    pred = pred.flatten()
    gt = gt.flatten()

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))

    dice = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)

    iou = (tp + epsilon) / (tp + fp + fn + epsilon)

    precision = (tp + epsilon) / (tp + fp + epsilon)

    recall = (tp + epsilon) / (tp + fn + epsilon)

    specificity = (tn + epsilon) / (tn + fp + epsilon)

    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity)
    }