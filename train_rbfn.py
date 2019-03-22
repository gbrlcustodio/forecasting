# -*- coding: utf-8 -*-
# Author:  Stefano Brilli
# Date:    24/10/2011
# E-mail:  stefanobrilli@gmail.com

import rbfn as network
import numpy as np
import numpy.linalg as la

def train(input, output, centers, gw=1.0):
    """
    Build a radial basis network
    input  (N by M) N vector of M size
    output (N by T) N vector of T size
    """
    k = np.sqrt(-np.log(0.5)) / gw
    G = ((centers[np.newaxis,:,:] - input[:, np.newaxis, :])**2.).sum(-1)
    G = np.exp(-( np.sqrt(G)*k )**2.0)
    W = la.lstsq(G, output)[0]
    return network.RBFN(centers=centers, ibias=k, linw=W, obias=0, gw=gw)

# if __name__ == "__main__":
#     # Simple test: recognising of points inside a ring
#     N = 3
#     I = (np.random.uniform(size=(N,2))-0.5)*2.
#     O = np.zeros((N,1))
#     O[ ((I**2.).sum(1) < 1)*((I**2.).sum(1) > 0.5)] = 1.0
#     r = train(I, O, 0.27)
#     err = abs(r.simulate(I) - O)

#     # The next error should be as much as possible closer to zero. Higher values
#     # may occur due to ill conditioned problems.
#     print("Error: ", max(np.sqrt((err**2.).sum(1))))
#     # Plot of some test value
#     import matplotlib.pyplot as plt
#     T = (np.random.uniform(size=(N*5,2))-0.5)*2
#     V = r.simulate(T).flatten()
#     OUT = T[V<0.5].T
#     IN = T[V>=0.5].T
#     plt.plot(IN[0], IN[1], 'r*', OUT[0], OUT[1], 'bo')
#     plt.show()
