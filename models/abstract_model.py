# -*- coding: utf-8 -*-
"""Abstract model definition."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class AbstractModel(nn.Module):

    def __init__(self, input_size):
        """
        Parameters
        ----------
        module : ModuleType
            Module defining the model.
        input_size : (int, int, int)
            Input size.
        """
        super(AbstractModel, self).__init__()
        self.input_size = input_size
        self.criterion = nn.MSELoss(reduction="sum")
        self.layers = None


    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input sample.

        Returns
        -------
        Tensor
            Output label.
        """
        if not self.layers:
            raise NotImplementedError

        self.kl = 0
        for layer in self.layers:
            out = layer(x)
            if len(out) == 2: # TODO: improve
                # Bayesian
                x, _kl = out
                self.kl += _kl
            else:
                # frequentist
                x = out
        return x.view(-1)


    def loss(self, pred, label, beta=0):
        """Compute loss.
        
        Parameters
        ----------
        pred : Tensor
            Predicted label.
        label : Tensor
            True label.
        beta : float, optional
            Beta factor.
        
        Returns
        -------
        Tensor
            Loss.
        """
        if not self.layers:
            raise NotImplementedError

        likelihood = -0.5 * self.criterion(pred, label) 
        complexity = beta * self.kl if beta != 0 else 0
        return complexity - likelihood


    def get_weight_statistics(self):
        """Extract weight statistics for later visualization (bias not used).
        
        Returns
        -------
        ([str], [ndarray], [ndarray])
            List of layer names,
            list of 1D array of `float` representing layer weight means,
            list of 1D array of `float` representing layer weight standard deviations.
        """
        if not self.layers:
            raise NotImplementedError

        names = []
        qmeans = []
        qstds = []
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "qw_mean") and hasattr(layer, "log_alpha"):
                names.append(str(layer.__class__).split(".")[-1].split("'")[0] + "-" + str(idx + 1))
                qmeans.append(layer.qw_mean.detach().cpu().numpy())
                qstds.append(np.sqrt(np.exp(layer.log_alpha.detach().cpu().numpy()) * (layer.qw_mean.detach().cpu().numpy() ** 2)))
            else:
                trainable_params = [param.detach().cpu().numpy() for param in layer.parameters() if param.requires_grad]
                if len(trainable_params) > 0:
                    weights = np.asarray(trainable_params[0])
                    names.append(str(layer.__class__).split(".")[-1].split("'")[0] + "-" + str(idx + 1))
                    qmeans.append(weights)
                    qstds.append(np.zeros(weights.shape))
        return names, qmeans, qstds


    # Source code modified from:
    # 	Title: sksq96/pytorch-summary
    # 	Author: Shubham Chandel (sksq96)
    # 	Date: 2018
    # 	Availability: https://github.com/sksq96/pytorch-summary/tree/b50f213f38544ac337beeeda93b03c7e48e69c78

    def summary(self, log_fn, batch_size=-1, device="cuda"):
        """Log model summary.
        
        Parameters
        ----------
        log_fn : callable
            Logging function.
        batch_size : int, optional
            Batch size.
        device : string, optional
            Device.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        if not self.layers:
            raise NotImplementedError

        def register_hook(module):

            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size

                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [[-1] + list(o.size())[1 :] for o in output][0]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "params_count") and callable(module.params_count):
                    params += module.params_count()
                    summary[m_key]["trainable"] = True
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == self)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ],  "Input device is not valid, please specify 'cuda' or 'cpu'."

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        #if isinstance(input_size, tuple):
            #input_size = [input_size]

        # batch_size of 2 for batchnorm
        #x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        x = torch.rand(1, *self.input_size).type(dtype)

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        self.apply(register_hook)

        # make a forward pass
        #self(*x)
        self(x)

        # remove these hooks
        for h in hooks:
            h.remove()

        log_fn("________________________________________________________________")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        log_fn(line_new)
        log_fn("================================================================")
        total_params = 0
        trainable_params = 0

        line_count = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]

            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            log_fn(line_new)
            line_count += 1

        log_fn("================================================================")
        log_fn("Total params: {0:,}".format(total_params))
        log_fn("Trainable params: {0:,}".format(trainable_params))
        log_fn("Non-trainable params: {0:,}".format(total_params - trainable_params))
        log_fn("________________________________________________________________")

        return trainable_params
