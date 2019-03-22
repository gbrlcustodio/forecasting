# -*- coding: utf-8 -*-
import numpy as np
from math import log

class RBFN(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def simulate(self, x):
        """
        Run the network over a single input and return the output value
        """
        v = np.atleast_2d(x)[:, np.newaxis]-self.centers[np.newaxis, :]
        v = np.sqrt( (v**2.).sum(-1) ) * self.ibias
        v = np.exp( -v**2. )
        v = np.dot(v, self.linw) + self.obias
        return v

    def aic(self, data, debug=None):
        outer_sum = 0
        for input, output in zip(data["INPUT"].values, data["OUTPUT"].values):
            inner_sum = 0
            for center, weight in zip(self.centers, self.linw):
                inner_sum += weight * self.gaussian(np.linalg.norm(input - center))

            if debug:
                print("Input: ", input, " Expected: ", output, ", Found: ", inner_sum)

            inner_sum = output - inner_sum
            outer_sum += inner_sum.T * inner_sum
            if debug:
                print("MSE: ", outer_sum/data["INPUT"].size)

        return data["INPUT"].size * log(outer_sum / data["INPUT"].size) + 4 * self.centers.size

    def gaussian(self, x):
        return np.exp(-(x * np.sqrt(-np.log(0.5)) / self.gw)**2.0)

    def _input_size(self):
        try:
            return self.centers.shape[1]
        except AttributeError:
            return -1
    input_size = property(_input_size)

    def _output_size(self):
        try:
            return self.linw.shape[1]
        except AttributeError:
            return -1
    output_size = property(_output_size)
