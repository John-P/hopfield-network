#!/usr/bin/env python3
from __future__ import print_function
from numpy import dot
from math import ceil, sqrt
from random import randint
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import colors

def connections(inhibition):
    """
    Connection weights as a 9*9 matrix dependent on an inhibition
    constant.
    """
    w = -inhibition
    weights = [[0, w, w, w, w, w, w, w, w],
               [w, 0, w, w, 1, w, w, 1, w],
               [w, w, 0, w, w, w, w, w, w],
               [w, w, w, 0, 1, 1, w, w, w],
               [w, 1, w, 1, 0, 1, w, 1, w],
               [w, w, w, 1, 1, 0, w, w, w],
               [w, w, w, w, w, w, 0, w, w],
               [w, 1, w, w, 1, w, w, 0, w],
               [w, w, w, w, w, w, w, w, 0]]
    return weights

def activate(x):
    """Threshold activation function"""
    return 1 if x > 0 else 0

def async(weights, state, *args):
    """
    Function whose output state at a randomly chosen unit is
    changed
    """
    a = randint(0, 8)
    c = [activate(x) for x in dot(weights, state)]
    state[a] = c[a]
    return state

def scan(weights, state):
    """Function which changes all unit states in scan order"""
    for i in range(len(state)):
        a = dot(weights, state)
        state[i] = activate(a[i])
    return state

#Table of initial states chosen at random
init =[[1, 1, 1, 0, 0, 1, 1, 0, 1],
       [1, 0, 0, 1, 0, 1, 0, 0, 1],
       [1, 1, 0, 0, 1, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 1, 0, 1, 1],
       [1, 0, 0, 1, 0, 0, 0, 0, 0],
       [1, 1, 0, 1, 1, 0, 1, 1, 1],
       [0, 0, 1, 0, 0, 0, 1, 1, 1],
       [1, 1, 0, 1, 0, 0, 0, 1, 0],
       [0, 1, 0, 1, 0, 1, 0, 0, 0],
       [0, 1, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 1, 0, 0, 0, 1],
       [0, 1, 0, 1, 1, 1, 0, 1, 1],
       [1, 0, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 0, 0, 1, 0, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 0, 0, 0],
       [1, 0, 0, 1, 1, 0, 0, 0, 1],
       [0, 0, 1, 0, 0, 0, 1, 0, 1],
       [1, 0, 0, 1, 1, 0, 1, 0, 1],
       [0, 0, 0, 0, 1, 0, 1, 1, 1],
       [1, 0, 0, 1, 1, 0, 0, 1, 0]]

def hopfield(f, weights, states, n):
    """
    Code for n cycles of an update function on table of
    initial states.
    """
    for _ in range(n):
        for s in states:
            s = f(weights, s)
    return states

def list_to_grid(a_list):
    """
    Takes a list and returns a square matrix (list of lists).
    Useful for plotting nodes.
    """
    side = ceil(sqrt(len(a_list)))
    grid = []
    while len(a_list) > 0:
        grid.append(a_list[:side])
        a_list = a_list[side:]
    return grid

def plot_state(state):
    """Plot an square image of a state"""
    # make a color map of fixed colors
    cmap = colors.ListedColormap(['white', 'black'])
    cmap.set_under('white')
    cmap.set_over('black')
    cmap.set_bad('red')
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    im_data = list_to_grid(state)
    im = plt.imshow(im_data, interpolation="nearest",
                    cmap=cmap, norm=norm)
    im.axes.get_xaxis().set_ticks([])
    im.axes.get_yaxis().set_ticks([])

    # make a color bar
    cbar = plt.colorbar(im, cmap=cmap, norm=norm,
                        boundaries=bounds, ticks=[0, 1])
    cbar.ax.tick_params(labelsize=30)
    plt.show()

def fixed_points(weights):
    """
    For a given weights table perform a scan update on each
    binary state vector to find fixed points
    """
    result = list()
    for state in product([0, 1], repeat=9):
        start = list(state)
        end = scan(weights, list(state))
        #If H(u) = u then it is a fixed point
        if end == start:
            result.append(end)
    return result

def energy(weights, state):
    """Compute the energy of the network for a given state"""
    return -0.5*sum([si*sj*weights[i][j]
                     for i, si in enumerate(state)
                     for j, sj in enumerate(state)])

def neighbours(weights, state):
    """
    Return a list of energies for a state's neighbours
    (state which differ by one bit).
    """
    result = []
    for i in range(len(state)):
        neighbour = state.copy()
        neighbour[i] ^= 1
        neighbour_energy = energy(weights, neighbour)
        result.append(neighbour_energy)
    return result

def plot_energy(f, weights, state, iterations, runs):
    def run(weights, state):
        """
        Perform updates of the network and return a list of
        energies for each iteration.
        """
        energies = []
        for i in range(iterations):
            energies.append(energy(weights, state))
            state = f(weights, state, i % len(state))
        return state, energies

    for i in range(1, runs+1):
        final_state, energies = run(weights, init[0].copy())
        line = plt.plot(energies, marker="x",
                        label="Run {}".format(i))
        arrowprops = {"arrowstyle":"->",
                      "connectionstyle":"arc3,rad=0"}
        plt.annotate(str(final_state),
                     xy=(len(energies)-1, energies[-1]),
                     xytext=(-10, -10),
                     textcoords="offset points", ha='right',
                     va='top', arrowprops=arrowprops)

    axes = plt.gca()
    axes.set_ylim([-5, 25])
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.title(str(init[0]))
    plt.legend()
    plt.show()

def update(weights, state, n):
    """
    Update the nth node in the network.
    """
    a = dot(weights, state)
    state[n] = activate(a[n])
    return state
