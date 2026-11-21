import numpy as np
from scipy.special import comb
import random
import matplotlib.pyplot as plt
import torch

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * t ** (n - i) * (1 - t) ** i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return (xvals, yvals)

def nonlinear_transformation_torch(x, flip, prob=0.5, num_free_coordinates=1, max_rnd_val=0.5, min_rnd_val=0):
    if random.random() >= prob:
        return x
    M, m = (x.max(), x.min())
    x = (x - m) / (M - m)
    assert num_free_coordinates in [1, 2, 4]
    rnd1 = random.random() * (max_rnd_val - min_rnd_val) + min_rnd_val
    rnd2 = random.random() * (max_rnd_val - min_rnd_val) + min_rnd_val
    rnd3 = random.random() * (max_rnd_val - min_rnd_val) + min_rnd_val
    rnd4 = random.random() * (max_rnd_val - min_rnd_val) + min_rnd_val
    if num_free_coordinates == 1:
        points = [[0, 0], [rnd1, 1 - rnd1], [1 - rnd1, rnd1], [1, 1]]
    elif num_free_coordinates == 2:
        points = [[0, 0], [rnd1, rnd2], [1 - rnd1, 1 - rnd2], [1, 1]]
    elif num_free_coordinates == 4:
        points = [[0, 0], [rnd1, rnd2], [rnd3, rnd4], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if flip:
        xvals = np.sort(xvals)
    else:
        xvals, yvals = (np.sort(xvals), np.sort(yvals))
    device = x.device
    dtype = x.dtype
    x = x.cpu().numpy()
    nonlinear_x = np.interp(x, xvals, yvals)
    nonlinear_x = torch.from_numpy(nonlinear_x).to(device=device, dtype=dtype)
    nonlinear_x = nonlinear_x * (M - m) + m
    return nonlinear_x
