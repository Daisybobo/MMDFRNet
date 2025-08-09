import torch
import torch.nn as nn

class DeepSupervisionLoss(nn.Module):
    """带深度监督的损失函数"""
    def __init__(self, weights=[1.0, 0.4, 0.3]):
        super().__init__()
        self.weights = weights
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            main_out, aux1, aux2 = outputs
            loss = self.weights[0] * self.criterion(main_out, target)
            loss += self.weights[1] * self.criterion(aux1, target)
            loss += self.weights[2] * self.criterion(aux2, target)
            return loss
        return self.criterion(outputs, target)