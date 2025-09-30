import numpy as np
from sklearn.metrics import (
    classification_report,
    hamming_loss
)
from typing import Dict

def order_combs(combs):
    final_combs = []
    np_combs = np.array(combs)
    for i in range(int(np.sqrt(len(combs)))+2):
        partial_combs = np_combs[np_combs.sum(axis=1) == i].astype(int).tolist()
        final_combs += sorted(partial_combs, reverse=True)
    
    return final_combs 

def binary_combinations(k):
    combs = []
    for i in range(2 ** k):
        # format number as binary string with k digits
        b = format(i, f'0{k}b')
        combs.append([int(x) for x in b])
    o_combs = order_combs(combs)
    return {str(comb): i for i, comb in enumerate(o_combs)}

def multilabel_eval(y_true: np.ndarray, y_pred: np.ndarray, logger, target_names=None, multi_task_target_names=None, print_metrics=True) -> Dict:
    """
    Evaluate a binary multi-label classification task.

    Args:
        y_true (np.ndarray): shape (n_samples, n_classes), ground truth.
        y_pred (np.ndarray): shape (n_samples, n_classes), predictions.
        target_names (list): optional, names of each class.

    Prints:
        - classification report per class
        - micro F1
        - macro F1
        - subset accuracy
        - hamming loss
    """
    n_classes = y_true.shape[1]
    if target_names is None:
        target_names = [f"Class {i}" for i in range(n_classes)]

    LABEL2INDEX = binary_combinations(n_classes)

    if print_metrics: logger.info("=== Per-class classification reports ===")
    metrics_report = {}
    for i in range(n_classes):
        metrics_report[target_names[i]] = classification_report(
            y_true[:, i],
            y_pred[:, i],
            zero_division=0,
            target_names=[f"Not {target_names[i]}", target_names[i]], 
            output_dict=True
        )
        if print_metrics:
            logger.info(
                classification_report(
                    y_true[:, i],
                    y_pred[:, i],
                    zero_division=0,
                    target_names=[f"Not {target_names[i]}", target_names[i]]
                    )
            )

    y_true_subset = [LABEL2INDEX[str(label)] for label in y_true.tolist()]
    y_pred_subset = [LABEL2INDEX[str(pred)] for pred in y_pred.tolist()]
    all_labels = list(range(len(multi_task_target_names)))  # e.g. [0,1,2,3]

    metrics_report['Subset'] = classification_report(
        y_true_subset,
        y_pred_subset,
        labels=all_labels, # in case the labels don't constitute all possible combinations                        
        target_names=multi_task_target_names,
        zero_division=0,
        output_dict=True
    )

    if print_metrics:
        logger.info("=== Subset classification report ===")
        logger.info(
            classification_report(
                y_true_subset,
                y_pred_subset,
                labels=all_labels,                        
                zero_division=0,
                target_names=multi_task_target_names
            )
        )

    # Hamming loss
    hamming = hamming_loss(y_true, y_pred)
    
    if print_metrics:
        logger.info(f"Hamming L. : {hamming:.3f}")
    
    metrics_report['es'] = {
        'val_loss': None,
        'ham_loss': hamming,
        'val_accuracy': metrics_report['Subset']['accuracy'],
        'val_macro_f1': metrics_report['Subset']['macro avg']['f1-score'],
        'val_weighted_f1': metrics_report['Subset']['weighted avg']['f1-score'],
    }
    return metrics_report
    
