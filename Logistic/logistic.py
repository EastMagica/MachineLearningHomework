#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2020/10/23 12:28
# @file     : logistic.py
# @project  : MachineLearning
# @software : PyCharm

import numpy as np


# Functions
# ---------

def logistic(x):
    return 1. / (1. + np.exp(-x))


def logistic_diff(x):
    return logistic(x) * (1 - logistic(x))


# Classes
# -------

class Logistic(object):
    def __init__(self, size, optimizer, iteration=100):
        self.iteration = iteration
        self.optimizer = optimizer
        self.size = size
        self.w = np.zeros(self.size)

    def train(self, x, y):
        x = np.hstack([
            x, np.zeros((x.shape[0], 1))
        ])
        self.w = np.random.standard_normal(self.size)
        cross_entropy_array = np.zeros(self.iteration)
        accuracy_array = np.zeros(self.iteration)
        for i in range(self.iteration):
            y_hat = self.predict(x)
            grad = (y_hat - y) @ x / x.shape[1]
            self.w += self.optimizer.delta(grad)
            cross_entropy_array[i] = np.mean(
                -y @ np.log(y_hat) - (1 - y) @ np.log(1 - y_hat)
            )
            accuracy_array[i] = np.mean(
                (y_hat > 0.5).astype(np.int) == y
            )
            print("cross_entropy: {:.2f}".format(cross_entropy_array[i]))
            print("accuracy: {:.2f}".format(accuracy_array[i]))
        return cross_entropy_array, accuracy_array

    def predict(self, x):
        return logistic(
            x @ self.w
        )
