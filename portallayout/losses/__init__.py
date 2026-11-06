"""
@date: 2021/7/19
@description:
"""

from torch.nn import L1Loss
# from torch.nn import BCELoss  # for opening
from torch.nn import BCEWithLogitsLoss  # for opening, and without using sigmoid
from portallayout.losses.led_loss import LEDLoss
from portallayout.losses.grad_loss import GradLoss
from portallayout.losses.boundary_loss import BoundaryLoss
from portallayout.losses.object_loss import ObjectLoss, HeatmapLoss
