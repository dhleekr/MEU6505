import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from manipulator_2d import Manipulator2D
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy

# model
# from stable_baselines import DDPG
# from stable_baselines import SAC
from stable_baselines import PPO2
# from stable_baselines import ACKTR
# from stable_baselines import A2C


# Call the robotic arm environment
env = Manipulator2D()

# model = DDPG.load('./models/ddpg/best_model')
# model = SAC.load('./models/sac/best_model')
model = PPO2.load('./0.001/best_model')
# model = ACKTR.load('./models/acktr/best_model')
# model = A2C.load('./models/a2c/best_model')


# while True:
#     obs = env.reset()
#     while True:
        
#         # The trained model returns action values using the current observation
#         action, _states = model.predict(obs)
#         obs, reward, done, info = env.step(action)
        
#         if done:
#             break

#     env.render()

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

env.render()
    



