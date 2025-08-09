import torch
import torch.optim as optim
import os
import wandb
from torch.multiprocessing import freeze_support
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np

from data_loader import get_data_loaders
from loss import DeepSupervisionLoss
from metrics import calculate_metrics
from MMDFRNet import MultiModalUNet

# set random seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# train
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    metrics_sum = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}

    for sar_data, optical_data, label in train_loader:
        sar_data, optical_data, label = (
            sar_data.to(device),
            optical_data.to(device),
            label.to(device)
        )

        optimizer.zero_grad()
        outputs = model(sar_data, optical_data)
        loss = criterion(outputs, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        if isinstance(outputs, tuple):
            pred = torch.argmax(outputs[0], dim=1)
        else:
            pred = torch.argmax(outputs, dim=1)
        batch_metrics = calculate_metrics(pred, label)

        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]


    avg_loss = running_loss / len(train_loader)
    avg_metrics = {k: v / len(train_loader) for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


# validate
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    metrics_sum = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}

    with torch.no_grad():
        for sar_data, optical_data, label in val_loader:
            sar_data, optical_data, label = (
                sar_data.to(device),
                optical_data.to(device),
                label.to(device)
            )

            outputs = model(sar_data, optical_data)
            loss = criterion(outputs, label)
            val_loss += loss.item()

            if isinstance(outputs, tuple):
                pred = torch.argmax(outputs[0], dim=1)
            else:
                pred = torch.argmax(outputs, dim=1)
            batch_metrics = calculate_metrics(pred, label)

            for k in metrics_sum:
                metrics_sum[k] += batch_metrics[k]

    avg_loss = val_loss / len(val_loader)
    avg_metrics = {k: v / len(val_loader) for k, v in metrics_sum.items()}

    return avg_loss, avg_metrics


def main():
    freeze_support()
    set_seed()
    wandb.init(project="rice-classification-mmf", name="simplified-unet-mmf")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root_dir = './Rice/training'
    batch_size = 16
    num_epochs = 60
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)

    # load data
    train_loader, val_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=batch_size,
        val_split=0.2,
        num_workers=4
    )

    model = MultiModalUNet(sar_channels=12, optical_channels=10, out_channels=2)
    model.to(device)
    criterion = DeepSupervisionLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': [], 'val_metrics': []
    }

    for epoch in range(num_epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['iou'])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)

    wandb.finish()


if __name__ == '__main__':

    main()
