# -*- coding: utf-8 -*-
"""BayesianConv5Dense1 definition. Sigmoid and softplus instead of tanh for positive output. No dropout."""

import torch.nn as nn

from utils.layers import BayesianConv2d, BayesianLinear, Flatten
from models.abstract_model import AbstractModel


class BayesianConv5Dense1(AbstractModel):

    def __init__(self, input_size):
        """
        Parameters
        ----------
        input_size : (int, int, int)
            Input size.
        """
        super(BayesianConv5Dense1, self).__init__(input_size)
       
        self.layers = nn.ModuleList([
            BayesianConv2d(input_size[0], 10, kernel_size=(10, 1), padding=(5, 0)),
            nn.Sigmoid(),
            BayesianConv2d(10, 10, kernel_size=(10, 1), padding=(4, 0)),
            nn.Sigmoid(),
            BayesianConv2d(10, 10, kernel_size=(10, 1), padding=(5, 0)),
            nn.Sigmoid(),
            BayesianConv2d(10, 10, kernel_size=(10, 1), padding=(4, 0)),
            nn.Sigmoid(),
            BayesianConv2d(10, 1, kernel_size=(3, 1), padding=(1, 0)),
            nn.Softplus(),
            Flatten(1 * input_size[1] * input_size[2]),
            BayesianLinear(1 * input_size[1] * input_size[2], 100),
            nn.Softplus(),
            BayesianLinear(100, 1),
            nn.Softplus()
        ])
