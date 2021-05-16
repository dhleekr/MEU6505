import numpy as np
from manipulator_2d import Manipulator2D
from Savebestcallback import SaveOnBestTrainingRewardCallback
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.bench import Monitor

# model 
# from stable_baselines import DDPG
from stable_baselines import SAC
# from stable_baselines import PPO2
# from stable_baselines import ACKTR
# from stable_baselines import A2C

# policy
# from stable_baselines.ddpg.policies import MlpPolicy
# from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.policies import MlpPolicy # ppo, acktr, a2c


n_timesteps = 20000000
log_dir = ''

# Call the robotic arm environment
env = Manipulator2D()
env = Monitor(env, 'log')

# Get how many actions do we need from the environment variable
n_actions = env.action_space.shape[-1]
param_noise = None

# Add exploration noise
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))


# Callback
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# Select algorithm from the stable-baseline library
# model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log="./ddpg/", gamma=0.9)
# model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./ppo2_gamma/",learning_rate=2.5e-4, gamma=0.9)
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./ppo2/',learning_rate=2.5e-4, gamma=0.9)
# model = SAC(MlpPolicy, env, verbose=1, action_noise=action_noise, tensorboard_log="./sac/", gamma=0.9, batch_size=256)
# model = ACKTR(MlpPolicy, env, verbose=1, tensorboard_log="./acktr/", learning_rate=0.25, gamma=0.9)
# model = A2C(MlpPolicy, env, verbose=1, tensorboard_log='./a2c/', learning_rate=0.0001, gamma=0.9)

# "learn" method calls "step" method in an environment
model.learn(total_timesteps=n_timesteps, callback=callback)

print('Success : {}'.format(env.count))
print('Out of boundary : {}'.format(env.out_count))

# Save the weights
# Additional assignment : How do I save the policy network that returned the best reward value during training, not the result of learning over 400,000 timesteps?
# Tip : Let's use the callback function for the learn method
# model.save("ppo2")