#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2020/10/23 9:21
# @file     : descent.py
# @project  : MachineLearning
# @software : PyCharm

import abc
import numpy as np


class GradientDescent(metaclass=abc.ABCMeta):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def delta(self, gradient):
        return -gradient * self.learning_rate


class Momentum(GradientDescent):
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.v = None
        self.beta = beta

    def delta(self, gradient):
        if self.v is None:
            self.v = np.zeros(gradient.shape)
        self.v = self.beta * self.v - self.learning_rate * gradient
        return self.v

