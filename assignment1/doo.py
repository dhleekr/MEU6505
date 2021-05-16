import gym
from gym import core, spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import math

def sum_angle(a, b):
    c = a + b
    if c >= np.pi:
        c -= 2 * np.pi
    elif c < -np.pi:
        c += 2 * np.pi
    return c

class Manipulator2D(gym.Env):
    
    def __init__(self, arm1=1, arm2=1, dt=0.01, tol=0.1):
        # Set the max and min value of the observation space.
        self.obs_high = np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]) # theta1, theta21, theta1d, theta21d
        self.obs_low = -self.obs_high

        # Set the max and min value of the action space.
        self.action_high = np.array([np.pi, np.pi])
        self.action_low = -self.action_high

        # These variables are required by the GYM environment, and these are the actual observation and action spaces.
        self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)
        self.action_space = spaces.Box(low = self.action_low, high = self.action_high, dtype = np.float32)

        # Variables for the robotic arm
        self.arm1 = arm1 # the length of arm 1
        self.arm2 = arm2 # the length of arm 2
        self.dt = dt # The size of timestep
        self.tol = tol # threshold; distance to the goal

        # Call reset method to reset the variables in the envrionment.
        self.reset()
        
        # Set the random seed
        self.seed()

        
    def step(self, action):
        # Clip the action value within the max/min value of the action space
        action = np.clip(action, self.action_low, self.action_high)

        # Computation part (Simulation)
        # Two actions; one for the angular velocity between the joint 1 and the base axis;
        # other for the angular velocity between the joint 2 and the end-effector
        self.theta1 += action[0] * self.dt
        self.theta21 += action[1] * self.dt
        self.theta2 = sum_angle(self.theta1, self.theta21)
        self.x1 = np.cos(self.theta1) * self.arm1
        self.y1 = np.sin(self.theta1) * self.arm1
        self.x2 = self.x1 + np.cos(self.theta2) * self.arm2
        self.y2 = self.y1 + np.sin(self.theta2) * self.arm2

        # Calulate the postions of the joints from the given angular velocities
        self.t += self.dt
        
        # Calculate the reward and get the terminal condition
        reward, done = self._get_reward(
            # The reward is calculated by the following eqn, which returns the distance to the goal
            np.linalg.norm([self.theta21d - self.theta21, self.theta1d - self.theta1])
        )

        # Additional information can be stored in this variable.
        info = {}

        # Store the position information of the robotic are for the visualization (rendering)
        
        self.buffer.append(
            [
                self.x1,
                self.y1,
                self.x2,
                self.y2,
                self.t,
                reward
            ]
        )

        # The "step" method in GYM environment returns
        # State(observation), reward at current step, terminal condtion of current episode, additional information
        return self._get_state(), reward, done, info


    def reset(self):
        # Called at the beginning of each episode
        # Reset the variables
        self.theta1 = 0
        self.theta21 = 0
        self.theta2 = sum_angle(self.theta21, self.theta1)

        # rd = 1.7
        # alphad = 3 * np.pi / 4

        # Assignment :Train the RL agent when the target point is not fixed and moves randomly in the first quadrant.
        # Tip : Try to change the state-space to make problem easier.
        rd = np.random.uniform(low=1.5, high=1.99)
        alphad = np.random.uniform(low=0, high=np.pi / 2)

        # Get the target point (goal)
        self.xd = rd * np.cos(alphad)
        self.yd = rd * np.sin(alphad)

        self.theta21d = math.acos((self.xd**2 + self.yd**2 - self.arm1**2 - self.arm2**2) / (2 * self.arm1 * self.arm2))
        self.theta1d = math.atan(self.yd / self.xd) - math.atan((self.arm2 * np.sin(self.theta21d)) / (self.arm1 + self.arm2 * np.cos(self.theta21d)))

        self.done = False
        self.t = 0
        self.buffer = [] # buffer for the visualization (render)

        # Unlike the "step" method, "reset" only returns the state(observation)
        return self._get_state()


    def _get_reward(self, l):
        # Calculate the reward and
        # terminal condition
        done = False

        # Set the target as a circle which has sqrt(self.tol) as a radius
        if l < self.tol: 
            reward = 10.
            # Terminate current episode if the end-effector gets near the target
            done = True 
        else:
            # The reward gets larger (in negative) if it is far from the target
            reward = -l**2

        return reward, done

    
    def _get_state(self):
        # Returns the state(observation)
        # Coordinates of joint2, end-effector, and the target point

        return np.array(
            [
                self.theta1,
                self.theta21,
                self.theta1d,
                self.theta21d
            ]
        )
    

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def render(self, plot_reward=False):
        # Plot the trajectory of the robotic arm within an episode

        buffer = np.array(self.buffer)
        plt.figure(1)
        plt.scatter([self.xd], [self.yd], c='r', marker='x', s=300)
        plt.plot(buffer[:, 0], buffer[:, 1], c='g')
        plt.plot(
            buffer[:, 2] - 0.1*np.cos(self.theta2),
            buffer[:, 3] - 0.1*np.sin(self.theta2),
            c='b'
        )
        plt.plot(
            [0, self.x1, self.x2 - 0.1*np.cos(self.theta2)],
            [0, self.y1, self.y2 - 0.1*np.sin(self.theta2)],
            marker='o',
            c='k'
        )
        plt.plot(
            [
                self.x2 + 0.1*np.cos(self.theta2) - 0.1*np.sin(self.theta2),
                self.x2 - 0.1*np.cos(self.theta2) - 0.1*np.sin(self.theta2),
                self.x2 - 0.1*np.cos(self.theta2) + 0.1*np.sin(self.theta2),
                self.x2 + 0.1*np.cos(self.theta2) + 0.1*np.sin(self.theta2)
            ],
            [
                self.y2 + 0.1*np.sin(self.theta2) + 0.1*np.cos(self.theta2),
                self.y2 - 0.1*np.sin(self.theta2) + 0.1*np.cos(self.theta2),
                self.y2 - 0.1*np.sin(self.theta2) - 0.1*np.cos(self.theta2),
                self.y2 + 0.1*np.sin(self.theta2) - 0.1*np.cos(self.theta2)
            ],
            c='k'
        )
        plt.axis('square')
        plt.title('Trajectory')
        if plot_reward:
            # Plot the reward curve
            plt.figure(2)
            plt.plot(buffer[:, 4], buffer[:, 5])
            plt.title('Rewards')
        plt.show()



def test(env):
    '''
    Test script for the environment "Manipulator2D"
    '''

    # Reset the environment
    env.reset()

    # Calculate the degree of the robotic arm to reach the target point
    c = (env.xd**2 + env.yd**2 - env.arm1**2 - env.arm2**2) / env.arm1 / 2
    s = np.sqrt(env.arm2**2 - c**2)
    theta21d = np.arctan2(s, c)
    theta1d = sum_angle(np.arctan2(env.yd, env.xd), - np.arctan2(s, env.arm1 + c))

    # Monitor the trajectories of the robotic arm for 10 seconds
    for t in np.arange(0, 10, env.dt):
        # Feedback control using the calculated degree value from the above
        action = [theta1d - env.theta1, theta21d - env.theta21]

        # Simulate for one timestep using the "step" method
        # Gets the changed state(observation), reward, terminal condition, and additional information
        next_state, reward, done, info = env.step(action)

        # Episode finished
        if done:
            break

    # Visualize the trajectories of the robotic arm
    env.render(plot_reward=True)


if __name__=='__main__':
    test(Manipulator2D(tol=0.01))