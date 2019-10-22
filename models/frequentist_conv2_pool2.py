# -*- coding: utf-8 -*-
"""FrequentistConv2Pool2 definition."""

import torch.nn as nn

from utils.layers import Flatten
from models.abstract_model import AbstractModel


class FrequentistConv2Pool2(AbstractModel):

    def __init__(self, input_size):
        """
        Parameters
        ----------
        input_size : (int, int, int)
            Input size.
        """
        super(FrequentistConv2Pool2, self).__init__(input_size)
       
        self.layers = nn.ModuleList([
            nn.Conv2d(input_size[0], 8, kernel_size=(5, 14), bias=False),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            nn.Conv2d(8, 14, kernel_size=(2, 1), bias=False),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 1)),
            Flatten(14 * int((((input_size[1] - 4) / 2) - 1) / 2) * (input_size[2] - 13)),
            nn.Linear(14 * int((((input_size[1] - 4) / 2) - 1) / 2) * (input_size[2] - 13), 1, bias=False)
        ])
