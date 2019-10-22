# -*- coding: utf-8 -*-
"""BayesianDense3 definition."""

import torch.nn as nn

from utils.layers import BayesianLinear, Flatten
from models.abstract_model import AbstractModel


class BayesianDense3(AbstractModel):

    def __init__(self, input_size):
        """
        Parameters
        ----------
        input_size : (int, int, int)
            Input size.
        """
        super(BayesianDense3, self).__init__(input_size)
      
        self.layers = nn.ModuleList([
            Flatten(input_size[0] * input_size[1] * input_size[2]),
            BayesianLinear(input_size[0] * input_size[1] * input_size[2], 100),
            nn.Sigmoid(),
            BayesianLinear(100, 100),
            nn.Sigmoid(),
            BayesianLinear(100, 100),
            nn.Sigmoid(),
            BayesianLinear(100, 1),
            nn.Softplus()
        ])
