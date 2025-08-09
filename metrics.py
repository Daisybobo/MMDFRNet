import torch

def calculate_metrics(pred, target):
    """计算分割任务的评价指标"""
    pred = pred.float()
    target = target.float()

    # IoU
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)

    # Precision
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    precision = (true_positive + 1e-8) / (predicted_positive + 1e-8)

    # Recall
    actual_positive = target.sum()
    recall = (true_positive + 1e-8) / (actual_positive + 1e-8)

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Accuracy
    correct = (pred == target).sum()
    total = target.numel()
    accuracy = correct / total

    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'accuracy': accuracy.item()
    }