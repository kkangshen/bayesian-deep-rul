# -*- coding: utf-8 -*-
"""BayesianConv2Pool2 definition. Softplus for positive output."""

import torch.nn as nn

from utils.layers import BayesianConv2d, BayesianLinear, Flatten
from models.abstract_model import AbstractModel


class BayesianConv2Pool2(AbstractModel):

    def __init__(self, input_size):
        """
        Parameters
        ----------
        input_size : (int, int, int)
            Input size.
        """
        super(BayesianConv2Pool2, self).__init__(input_size)
       
        self.layers = nn.ModuleList([
            BayesianConv2d(input_size[0], 8, kernel_size=(5, 14)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            BayesianConv2d(8, 14, kernel_size=(2, 1)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            Flatten(14 * int((((input_size[1] - 4) / 2) - 1) / 2) * (input_size[2] - 13)),
            BayesianLinear(14 * int((((input_size[1] - 4) / 2) - 1) / 2) * (input_size[2] - 13), 1),
            nn.Softplus()
        ])
