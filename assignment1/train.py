import numpy as np
from robotarm_env import Manipulator2D
# from doo import Manipulator2D
from Savebestcallback import SaveOnBestTrainingRewardCallback

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.bench import Monitor

n_timesteps = 400000
log_dir = '/mnt/c/Users/Owner/Desktop/class/optimal control and reinforcement learning/assignment1/'

# Call the robotic arm environment
env = Manipulator2D()
env = Monitor(env, log_dir)

# Get how many actions do we need from the environment variable
n_actions = env.action_space.shape[-1]
param_noise = None

# Add exploration noise
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))


# Callback
# eval_callback = EvalCallback(env, best_model_save_path='/mnt/c/Users/Owner/Desktop/class/optimal control and reinforcement learning/assignment/')
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Select DDPG algorithm from the stable-baseline library
model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)


# Train an RL agent for 400,000 timesteps
# "learn" method calls "step" method in an environment
model.learn(total_timesteps=n_timesteps, callback=callback)

# Save the weights
# Additional assignment : How do I save the policy network that returned the best reward value during training, not the result of learning over 400,000 timesteps?
# Tip : Let's use the callback function for the learn method
model.save("ddpg_manipulator2D")

# Delete the model variable from the memory
del model # remove to demonstrate saving and loading

""" # Load the weights from the saved training file
model = DDPG.load("ddpg_manipulator2D")

# Reset the simulation environment
obs = env.reset()

while True:
    # The trained model returns action values using the current observation
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    
    if done:
        break

env.render(True) """