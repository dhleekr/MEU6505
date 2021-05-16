from solvers.policy_iteration import PolicyIteration
from solvers.value_iteration import ValueIteration
from examples.distance_keeping_control import constructMDP
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":

    MDP = constructMDP()
    solver = PolicyIteration(MDP)
    # solver = ValueIteration(MDP)

    sol = solver.solve(
        max_iteration = 10,
        tolerance = 1e-10,
        verbose = True,
        logging = False
    )

    values = sol['values']
    policy = sol['policy']

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # plt.imshow(policy.reshape((-1,361)))
    # plt.xlabel(r'$\alpha$', fontsize=16)
    # plt.ylabel(r'$r$', fontsize=16)
    plt.figure(1)
    plt.imshow(values.reshape((-1,361)))
    plt.xlabel(r'$\alpha$', fontsize=16)
    plt.ylabel(r'$r$', fontsize=16)
    plt.figure(2)
    plt.imshow(policy.reshape((-1,361)))
    plt.xlabel(r'$\alpha$', fontsize=16)
    plt.ylabel(r'$r$', fontsize=16)
    
    plt.show()
    