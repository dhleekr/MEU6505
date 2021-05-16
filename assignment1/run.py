import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from robotarm_env import Manipulator2D
# from doo import Manipulator2D

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy

log_dir = '/mnt/c/Users/Owner/Desktop/class/optimal control and reinforcement learning/assignment/'



def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


n_timesteps = 400000

# Call the robotic arm environment
env = Manipulator2D()

model = DDPG.load('best_model')

# Reset the simulation environment
obs = env.reset()

while True:
    # The trained model returns action values using the current observation
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render(True)
    
    if done:
        break


env.render(True)


# results_plotter.plot_results(['/mnt/c/Users/Owner/Desktop/class/optimal control and reinforcement learning/assignment/'], n_timesteps, results_plotter.X_TIMESTEPS, "DDPG 2D 2dof robot arm")

# plot_results(log_dir)

plt.show()