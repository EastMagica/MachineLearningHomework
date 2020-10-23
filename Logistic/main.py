#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2020/10/22 23:45
# @file     : main.py
# @project  : MachineLearning
# @software : PyCharm

import numpy as np
import matplotlib.pyplot as plt

from descent import GradientDescent, Momentum


# Functions
# ---------

def load_data(filename="data.csv"):
    return np.loadtxt(
        filename, delimiter=","
    )


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


if __name__ == "__main__":
    data = load_data()
    lr = Logistic(
        size=data.shape[1],
        optimizer=Momentum(),
        iteration=100
    )
    cross_entropy, accuracy = lr.train(
        data[:, :-1], data[:, -1]
    )

    fig, ax = plt.subplots(2, 1, figsize=(9, 6))

    ax[0].plot(cross_entropy, marker=".")
    ax[0].set_title("Cross Entropy")
    ax[0].set_xlabel("Iteration")

    ax[1].plot(accuracy, marker=".")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Iteration")

    plt.tight_layout()
    plt.show()

