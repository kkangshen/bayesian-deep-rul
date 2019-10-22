# -*- coding: utf-8 -*-
"""Dataloader."""

# Source code modified from:
# 	Title: charlesq34/pointnet2
# 	Author: Charles R. Qi (charlesq34)
# 	Date: 2017
# 	Availability: https://github.com/charlesq34/pointnet2/tree/42926632a3c33461aebfbee2d829098b30a23aaa

import os

import numpy as np
np.random.seed(seed=42)


class Dataloader():

    def __init__(self, root="./", dataset="CMAPSS/FD001", split="train", normalization="min-max", batch_size=512, channel_first=True, shuffle=None, max_rul=10000, quantity=1.0, cache_size=100000):
        """
        Parameters
        ----------
        root : str, optional
            Root directory of the dataset.
        dataset : str, optional
            Dataset to load.
        split : str, optional
            Portion of the data to load. Either 'train' or 'validation' or 'test'.
        normalization : str, optional
            Normalization strategy. Either 'min-max' or 'z-score'.
        batch_size : int, optional
            Batch size.         
        channel_first : bool, optional
            True to load channels in the first dimension, False to load channels in the last one.
        shuffle: bool or None, optional
            True to shuffle the samples, False otherwise. If None, only training samples are shuffled by default.
        max_rul : int, optional
            Label rectification threshold.
        quantity: float, optional
            Ratio of data to use. (1 - quantity) ratio of samples are randomly dropped.
        cache_size: int, optional
            Number of samples to cache.
        """
        assert split in ["train", "validation", "test"], "'split' must be either 'train' or 'validation' or 'test', got '" + split + "'."

        assert normalization in ["z-score", "min-max"], "'normalization' must be either 'z-score' or 'min-max', got '" + normalization + "'."

        assert 0 <= quantity <= 1.0, "'quantity' must be a value within [0, 1], got %.2f" % quantity + "."

        self.root = root
        tmp = dataset.split("/")
        self.dataset = tmp[0] + "/data" if len(tmp) == 1 else tmp[0] + "/data/" + tmp[1]
        self.split = split
        self.normalization = normalization
        self.batch_size = batch_size
        self.channel_first = channel_first
        self.max_rul = max_rul

        # list of file_path
        self.datapath = [os.path.join(self.root, self.dataset, self.normalization, self.split, file_name) \
                        for file_name in os.listdir(os.path.join(self.root, self.dataset, self.normalization, self.split))] \
                        if os.path.exists(os.path.join(self.root, self.dataset, self.normalization, self.split)) \
                        else []

        # remove random elements from the list
        original_length = len(self.datapath)
        while len(self.datapath) > round(original_length * quantity):
            del self.datapath[np.random.randint(low=0, high=len(self.datapath))]

        # init data shape
        self.num_channels = 0
        self.window = 0
        self.num_features = 0

        # get data shape
        if len(self.datapath) > 0:
            with open(self.datapath[0], "r") as _:
                lines = _.readlines()
                self.num_channels = 1
                self.window = len(lines)
                self.num_features = len(lines[0].split())

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (sample, label) tuple

        self.shuffle = shuffle or (split == "train")

        self.reset()


    def _get_item(self, index):
        """Retrieve sample and label from either cache or source file.
        
        Parameters
        ----------
        index : int
            Sample index.
        
        Returns
        -------
        (ndarray, ndarray)
            4D array of `float` representing the data sample, 1D array of `float` (of size = 1) representing the label.
        """
        if index in self.cache:
            sample, label = self.cache[index]
        else:
            fn = self.datapath[index]
            label = float(fn.split("-")[-1].replace(".txt", ""))
            # rectify label
            if label > self.max_rul:
                label = self.max_rul
            label = np.array([label]).astype(np.float32)
            sample = np.loadtxt(fn).astype(np.float32)
            if self.channel_first:
                sample = sample.reshape(-1, self.num_channels, self.window, self.num_features)
            else:
                sample = sample.reshape(-1, self.window, self.num_features, self.num_channels)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (sample, label)
        return sample, label
        

    def __getitem__(self, index):
        """Retrieve sample and label from either cache or source file.
        
        Parameters
        ----------
        index : int
            Sample index.
        
        Returns
        -------
        (ndarray, ndarray)
            4D array of `float` representing the data sample, 1D array of `float` (of size = 1) representing the label.
        """
        return self._get_item(index)


    def __len__(self):
        """Return total number of samples.
        
        Returns
        -------
        int
            Total number of samples.
        """
        return len(self.datapath)


    def reset(self):
        """Reset batch index."""
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath) + self.batch_size - 1) // self.batch_size
        self.batch_idx = 0


    def has_next_batch(self):
        """        
        Returns
        -------
        bool
            True if a next batch exists, False otherwise.
        """
        return self.batch_idx < self.num_batches


    def next_batch(self):
        """Retrieve next batch of samples and labels.
               
        Returns
        -------
        (ndarray, ndarray)
            4D array of `float` representing the next batch of data samples, 1D array of `float` representing the next batch of labels.
        """
        # returned dimension may be smaller than self.batch_size
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        if self.channel_first:
            batch_data = np.zeros((bsize, self.num_channels, self.window, self.num_features))
        else:
            batch_data = np.zeros((bsize, self.window, self.num_features, self.num_channels))
        batch_label = np.zeros((bsize), dtype=np.float32)
        for i in range(bsize):
            sample, label = self._get_item(self.idxs[i + start_idx])
            batch_data[i] = sample
            batch_label[i] = label
        self.batch_idx += 1
        return batch_data, batch_label
