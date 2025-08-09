import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset_enhanced import MyDataset


def get_data_loaders(root_dir, batch_size=16, val_split=0.2, num_workers=4):
    """获取训练和验证数据加载器"""
    # 加载数据集
    train_dataset = MyDataset(root_dir, train=True)
    val_dataset = MyDataset(root_dir, train=False)

    # 数据集分割
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int((1 - val_split) * dataset_size)
    train_indices, val_indices = indices[:split], indices[split:]

    # 创建采样器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers
    )

    return train_loader, val_loader