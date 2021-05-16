import gym
import os
import matplotlib.pyplot as plt

# Example RL Alogirthm
from stable_baselines import PPO2

# 2D Arm Env
from manipulator_2d import Manipulator2D

# Create environment
# env = Manipulator2D()

# Load the trained agent
# Load the weight file in relative path.
model1 = PPO2.load('./models/best_model(0.1)')
model2 = PPO2.load('./models/best_model(0.01)')
model3 = PPO2.load('./models/best_model(0.005)')
model4 = PPO2.load('./models/best_model(0.001)')

# Competition Code
# The code below will be overwritten by the TA.
# tol = 0.1 
a_num = []
a_time = []
a_numsum = 0
a_timesum = 0
for j in range(1000):
    env = Manipulator2D()
    obs = env.reset()
    print('a({})'.format(j))
    for i in range(1000):
        action, _states = model1.predict(obs)
        obs, rewards, dones, info = env.step(action)
    a_num.append(env.count)
    a_numsum += env.count

for j in range(1000):
    env = Manipulator2D()
    obs = env.reset()
    print('a({})'.format(j))
    for i in range(1000):
        action, _states = model1.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones == True:
            a_time.append(env.t)
            a_timesum += env.t
            break
        

# tol = 0.01
b_num = []
b_time = []
b_numsum = 0
b_timesum = 0
for j in range(1000):
    env = Manipulator2D()
    obs = env.reset()
    print('b({})'.format(j))
    for i in range(1000):
        action, _states = model2.predict(obs)
        obs, rewards, dones, info = env.step(action)
    b_num.append(env.count)
    b_numsum += env.count

for j in range(1000):
    env = Manipulator2D()
    obs = env.reset()
    print('b({})'.format(j))
    for i in range(1000):
        action, _states = model2.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones == True:
            b_time.append(env.t)
            b_timesum += env.t
            break

# tol = 0.005
c_num = []
c_time = []
c_numsum = 0
c_timesum = 0
for j in range(1000):
    env = Manipulator2D()
    obs = env.reset()
    print('c({})'.format(j))
    for i in range(1000):
        action, _states = model3.predict(obs)
        obs, rewards, dones, info = env.step(action)
    c_num.append(env.count)
    c_numsum += env.count

for j in range(1000):
    env = Manipulator2D()
    obs = env.reset()
    print('c({})'.format(j))
    for i in range(1000):
        action, _states = model3.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones == True:
            c_time.append(env.t)
            c_timesum += env.t
            break

# tol = 0.001
d_num = []
d_time = []
d_numsum = 0
d_timesum = 0
for j in range(1000):
    env = Manipulator2D()
    obs = env.reset()
    print('d({})'.format(j))
    for i in range(1000):
        action, _states = model4.predict(obs)
        obs, rewards, dones, info = env.step(action)
    d_num.append(env.count)
    d_numsum += env.count

for j in range(1000):
    env = Manipulator2D()
    obs = env.reset()
    print('d({})'.format(j))
    for i in range(1000):
        action, _states = model4.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones == True:
            d_time.append(env.t)
            d_timesum += env.t
            break


print('0.1 num mean : {}\t0.01 num mean : {}\t0.005 num mean : {}\t0.001 num mean : {}'.format(a_numsum/1000, b_numsum/1000, c_numsum/1000, d_numsum/1000))
print('0.1 time mean : {}\t0.01 time mean : {}\t0.005 time mean : {}\t0.001 time mean : {}'.format(a_timesum/1000, b_timesum/1000, c_timesum/1000, d_timesum/1000))

x = [i for i in range(1000)]
plt.figure(1)
plt.plot(x, a_num, label='tol=0.1')
plt.plot(x, b_num, label='tol=0.01')
plt.plot(x, c_num, label='tol=0.005')
plt.plot(x, d_num, label='tol=0.001')

plt.legend()
plt.title('Compare models')
plt.xlabel('Episodes')
plt.ylabel('Success')

plt.figure(2)
plt.plot(x, a_time, label='tol=0.1')
plt.plot(x, b_time, label='tol=0.01')
plt.plot(x, c_time, label='tol=0.005')
plt.plot(x, d_time, label='tol=0.001')

plt.legend()
plt.title('Compare models')
plt.xlabel('Episodes')
plt.ylabel('Time')

plt.show()

# env.render()