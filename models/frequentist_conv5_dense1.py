# -*- coding: utf-8 -*-
"""FrequentistConv5Dense1 definition. Last tanh removed due to vanishing gradient."""

import torch.nn as nn

from utils.layers import Flatten
from models.abstract_model import AbstractModel


class FrequentistConv5Dense1(AbstractModel):

    def __init__(self, input_size):
        """
        Parameters
        ----------
        input_size : (int, int, int)
            Input size.
        """
        super(FrequentistConv5Dense1, self).__init__(input_size)
       
        self.layers = nn.ModuleList([
            nn.Conv2d(input_size[0], 10, kernel_size=(10, 1), padding=(5, 0), bias=False),
            nn.Tanh(),
            nn.Conv2d(10, 10, kernel_size=(10, 1), padding=(4, 0), bias=False),
            nn.Tanh(),
            nn.Conv2d(10, 10, kernel_size=(10, 1), padding=(5, 0), bias=False),
            nn.Tanh(),
            nn.Conv2d(10, 10, kernel_size=(10, 1), padding=(4, 0), bias=False),
            nn.Tanh(),
            nn.Conv2d(10, 1, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.Tanh(),
            Flatten(1 * input_size[1] * input_size[2]),
            nn.Dropout(0.5),
            nn.Linear(1 * input_size[1] * input_size[2], 100, bias=False),
            nn.Linear(100, 1, bias=False)
            #nn.Tanh()
        ])

        def weights_init(m):
            """Xavier initialization.

            Parameters
            ----------
            m : Module
                Layer.
            """
            classname = m.__class__.__name__
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                # Xavier initialization not defined for scalar values
                #nn.init.xavier_normal_(m.bias.data)

        self.apply(weights_init)
