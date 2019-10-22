# -*- coding: utf-8 -*-
"""FrequentistDense3 definition."""

import torch.nn as nn

from utils.layers import Flatten
from models.abstract_model import AbstractModel


class FrequentistDense3(AbstractModel):

    def __init__(self, input_size):
        """
        Parameters
        ----------
        input_size : (int, int, int)
            Input size.
        """
        super(FrequentistDense3, self).__init__(input_size)
       
        self.layers = nn.ModuleList([
            Flatten(input_size[0] * input_size[1] * input_size[2]),
            nn.Linear(input_size[0] * input_size[1] * input_size[2], 100, bias=False),
            nn.Sigmoid(),
            nn.Linear(100, 100, bias=False),
            nn.Sigmoid(),
            nn.Linear(100, 100, bias=False),
            nn.Sigmoid(),
            nn.Linear(100, 1, bias=False)
        ])
