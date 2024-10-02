import os
from pathlib import Path
import numpy as np
import pandas as pd

from utils.settings.config import RANDOM_SEED
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate




np.random.seed(RANDOM_SEED)
def plot_save_confusion_matrix(cm, ticks, fontsize=16, norm = False, save_path = None):
    if norm:
        cm = cm / cm.sum(axis=1, keepdims=True)
    # show the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', annot_kws={"size": fontsize})
    # set x label
    plt.xlabel('Predicted labels', fontsize=fontsize)
    # set y label
    plt.ylabel('True labels', fontsize=fontsize)
    # set xticks
    if ticks is not None:
        tick_positions = np.arange(len(ticks)) + 0.5
        plt.xticks(tick_positions, ticks, rotation=90, fontsize=fontsize)
        plt.yticks(tick_positions, ticks, rotation=0, fontsize=fontsize)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
def compute_metrics(cm):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    num_classes = cm.shape[0]
    TN = []
    for i in range(num_classes):
        temp = np.delete(cm, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*(precision*recall)/(precision+recall)
    cls_accuracy = (TP+TN)/cm.sum()

    overall_accuracy = np.trace(cm) / np.sum(cm)
    return TP, FP, FN, TN, specificity, precision, recall, f1, cls_accuracy, overall_accuracy
def update_confusion_matrix( confusion_matrix, label, pred):
    h,w = confusion_matrix.size()
    num_classes = confusion_matrix.size(0)
    indices = label * num_classes + pred
    values = torch.ones_like(label, dtype=torch.float32)
    return confusion_matrix.view(-1).index_add_(0, indices.to(confusion_matrix.device).to(torch.int32), values.to(confusion_matrix.device)).view(h, w)
def on_test_epoch_end( run_path, checkpoint_epoch, confusion_matrix, label_encoder):
    np.save(run_path / f'cm_epoch{checkpoint_epoch}.npy', confusion_matrix)

    TP, FP, FN, TN, specificity, precision, recall, f1, cls_accuracy, overall_accuracy = compute_metrics(confusion_matrix)
    


    confusion_matrix = np.nan_to_num(confusion_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Create plot
    confusion_matrix_norm = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # swap key and values
    label_encoder = {v: k for k, v in label_encoder.items()}

    ticks = [label_encoder[i] for i in range(len(label_encoder))]

    save_path = run_path / f'confusion_matrix_norm_epoch{checkpoint_epoch}.pdf'

    plot_save_confusion_matrix(confusion_matrix_norm, ticks, fontsize=12,save_path = save_path)
    
    # just for printing
    metrics = {
    'TP': TP,
    'FP': FP,
    'FN': FN,
    'TN': TN,
    # 'Specificity': specificity,
    'Precision': precision,
    'Recall': recall,
    'F1': f1,
    'Overall_Accuracy': [overall_accuracy] + [np.nan] * (len(TP) - 1),
    # 'Class Accuracy': cls_accuracy
    }
    metrics_df = pd.DataFrame(metrics, index=ticks)
    metrics_df = metrics_df.T
    metrics_df.to_csv(run_path / f"evaluation_metrics_epoch{checkpoint_epoch}.csv", float_format='%.3f')
    print(tabulate(metrics_df, headers='keys', tablefmt='grid'))
def NVDI(B04, B08):
    # https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index
    # NVDI = (NIR - R) / (NIR + R)
    return (B08 - B04) / (B08 + B04)
