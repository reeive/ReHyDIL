import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
    def _dice_loss(self, score, target):
        smooth = 1e-6
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice_loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice_loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        dice_loss = self._dice_loss(inputs, target)
        loss = 1 - dice_loss
        return loss           

def logits_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes logits on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert input_logits.size() == target_logits.size()
    mse_loss = (input_logits-target_logits)**2
    return mse_loss