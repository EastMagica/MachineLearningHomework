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
from logistic import Logistic


# Functions
# ---------

def load_data(filename="data.csv"):
    return np.loadtxt(
        filename, delimiter=","
    )


def plot(cross_entropy, accuracy):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))

    ax[0].plot(cross_entropy, marker=".")
    ax[0].set_title("Cross Entropy")
    ax[0].set_xlabel("Iteration")

    ax[1].plot(accuracy, marker=".")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Iteration")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_data()
    lr = Logistic(
        size=data.shape[1],
        # optimizer=GradientDescent(
        #     learning_rate=0.5
        # ),
        optimizer=Momentum(
            learning_rate=0.5
        ),
        iteration=100
    )
    ce, ac = lr.train(
        data[:, :-1], data[:, -1]
    )

    plot(ce, ac)

