import gym
from gym import core, spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

from ou_noise import OUNoise



class Transformation:
    """
    Transformation class for SE(2)
    """

    def __init__(self, matrix=None, translation=(0, 0), rotation=0):
        if isinstance(matrix, None.__class__):
            self._matrix = self.compute_matrix(translation, rotation)
        else:
            self._matrix = matrix.copy()
    
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            tmp = Transformation()
            tmp._matrix = np.matmul(self._matrix, other._matrix)
            return tmp
        elif isinstance(other, np.ndarray):
            if other.shape==(2,):
                return np.matmul(self._matrix, np.concatenate((other, [1])))[:2]
            else:
                return np.matmul(self._matrix, other)
        else:
            return self._matrix * other

    def __str__(self):
        return "Translation: %s\nRotation: %s\nTransfromation matrix:\n%s"%(
            self.get_translation(), self.get_rotation(), self._matrix
        )

    def transform(self, translation=(0, 0), rotation=0):
        self._matrix = np.matmul(self._matrix, self.compute_matrix(translation, rotation))

    def reset(self):
        self._matrix = self.compute_matrix((0, 0), 0)

    def get_translation(self):
        return self._matrix[0:2, 2]

    def get_rotation(self):
        return self._matrix[0:2, 0:2]

    def get_transformation(self):
        return self._matrix

    def x(self, x=None):
        if isinstance(x, None.__class__):
            return self._matrix[0, 2]
        else:
            self._matrix[0, 2] = x

    def y(self, y=None):
        if isinstance(y, None.__class__):
            return self._matrix[1, 2]
        else:
            self._matrix[1, 2] = y

    def euler_angle(self, angle=None):
        if isinstance(angle, None.__class__):
            return np.arctan2(self._matrix[1, 0], self._matrix[0, 0])
        elif isinstance(angle, float):
            self._matrix[0:2, 0:2] = self.compute_matrix((0, 0), angle)[0:2, 0:2]

    def inv(self, return_class=True):
        if return_class:
            tmp = Transformation()
            tmp._matrix = np.linalg.inv(self._matrix)
            return tmp
        else:
            return np.linalg.inv(self._matrix)

    def copy(self):
        tmp = Transformation(matrix=self._matrix)
        return tmp

    @staticmethod
    def compute_matrix(translation, rotation): # 2?????? rotation + translation
        c = np.cos(rotation)
        s = np.sin(rotation)
        return np.array(
            [
                [c, -s, translation[0]],
                [s,  c, translation[1]],
                [0,  0,              1]
            ]
        )



class Manipulator2D(gym.Env):
    
    def __init__(self, arm1=1, arm2=1, dt=0.01, tol=0.1):
        self.env_boundary = 5
        self.count = 0
        self.out_count = 0

        # Observation space??? ???????????? state??? ??????, ????????? ????????????.
        self.obs_high = np.array([
            self.env_boundary, self.env_boundary,
            self.env_boundary + arm1, self.env_boundary + arm1,
            self.env_boundary + arm2, self.env_boundary + arm2,
            self.env_boundary, self.env_boundary,
        ])
        self.obs_low = -self.obs_high
 
        # Action space??? ???????????? action??? ??????, ????????? ????????????.
        self.action_high = np.array([1, np.pi*2, np.pi*2, np.pi])
        self.action_low = np.array([0, -np.pi*2, -np.pi*2, -np.pi])

        # GYM environment?????? ???????????? ?????????, ?????? observation space??? action space??? ???????????? ????????????.
        self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)
        self.action_space = spaces.Box(low = self.action_low, high = self.action_high, dtype = np.float32)

        # ???????????? ????????? ???????????? ??????
        self.link1_len = arm1 # ????????? ??????
        self.link2_len = arm2
        self.dt = dt # Timestep
        self.tol = tol # ???????????? ??????

        # ?????? ???????????? ????????? ?????? ????????? ????????? seed??? ????????????.
        self.seed()

        self.target_speed = 1.2

        # ????????? ???????????????.
        self.reset()

        
    def step(self, action):
        self._move_target()

        # ??????????????? ???????????? action??? ????????? ????????? ??????, ?????? action?????? ???????????????.
        action = np.clip(action, self.action_low, self.action_high)

        # Action???????????? ????????? kinematics??? ???????????? ??????
        # ???????????? action??? ??? ?????????1??? x?????? ????????? ?????? ??????, ?????????1??? ?????????2??? ????????? ?????? ????????? ??????
        self.robot_tf.transform(
            translation=(action[0]*self.dt, 0),
            rotation=action[1]*self.dt
        )

        self.joint1_tf.transform(rotation=action[2] * self.dt)
        self.joint2_tf.transform(rotation=action[3] * self.dt)

        self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
        self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf

        self.t += self.dt

        # Reward??? episode ?????? ????????? ??????
        reward, done = self._get_reward()

        # ?????? ???????????? ????????? ???????????? ?????????
        info = {}

        # ????????? ???????????? ????????? ???????????? self.buffer ??? ??????
        self.buffer.append(
            dict(
                robot=self.robot_tf.copy(),
                link1=self.link1_tf_global.copy(),
                link2=self.link2_tf_global.copy(),
                target=self.target_tf.copy(),
                time=self.t,
                reward=reward
            )
        )

        # ??????????????? Gym environment??? step function??? 
        # State(observation), ?????? step????????? reward, episode ?????? ??????, ?????? ????????? ??????????????????
        return self._get_state(), reward, done, info


    def reset(self):
        # ??? episode??? ???????????? ?????????.
        # ?????? ????????? ?????????
        self.robot_tf = Transformation()
        self.joint1_tf = Transformation()
        self.link1_tf = Transformation(translation=(self.link1_len, 0))
        self.joint2_tf = Transformation()
        self.link2_tf = Transformation(translation=(self.link2_len, 0))
        self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
        self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf

        # ?????? ?????? ??????
        self.target_tf = Transformation(
            translation=(
                random.randrange(-self.env_boundary, self.env_boundary),
                random.randrange(-self.env_boundary, self.env_boundary)
            )
        )
        self.ou = OUNoise(dt=self.dt, theta=0.1, sigma=0.2) # ????????? 0.2??????

        self.done = False
        self.t = 0
        self.buffer = []    # ???????????? ?????? ??????. episode??? ????????? ????????? ?????????.

        # Step ????????? ????????? reset????????? ?????? state ??? ?????? ???????????????.
        return self._get_state()


    def _move_target(self):
        self.target_tf.transform(
            translation = (self.target_speed * self.dt, 0),
            rotation = self.ou.evolve() * self.dt
        )
        if self.target_tf.x() > self.env_boundary:
            self.target_tf.x(self.env_boundary)
        if self.target_tf.x() < -self.env_boundary:
            self.target_tf.x(-self.env_boundary)
        if self.target_tf.y() > self.env_boundary:
            self.target_tf.y(self.env_boundary)
        if self.target_tf.y() < -self.env_boundary:
            self.target_tf.y(-self.env_boundary)


    def _get_reward(self):
        done = False

        l = np.linalg.norm(
            self.target_tf.get_translation() - self.link2_tf_global.get_translation()
        )

        if l < self.tol:
            self.count += 1
            print("Success!!!!!!!")
            print('time : {} \tcount : {}'.format(self.t, self.count))
            reward = 1000
            done = True
        else:
            # timestep?????? (-)reward??? ?????? minimum time control??? ?????? ???.
            reward = -l**2

        x0, y0 = self.robot_tf.get_translation()
        if abs(x0) > self.env_boundary:
            self.out_count += 1
            print("Robot??? Boundary??? ?????????.")
            print('count : {}'.format(self.out_count))
            reward = -3000
            done = True
            
        elif abs(y0) > self.env_boundary:
            self.out_count += 1
            print("Robot??? Boundary??? ?????????.")
            print('count : {}'.format(self.out_count))
            reward = -3000
            done = True

        if self.t > self.dt * 1000:
            print("Time over")
            reward = -3000
            done = True
        
        return reward, done

    
    def _get_state(self):
        # State(Observation)??? ???????????????.

        link2_to_target = self.link2_tf_global.inv() * self.target_tf.get_translation()
        err1 = self.link2_tf * link2_to_target
        err2 = self.link1_tf * self.joint2_tf * err1
        err3 = self.joint1_tf * err2

        return np.concatenate(
            [
                link2_to_target,
                err1,
                err2,
                err3
            ]
        )

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def render(self):
        # Episode ????????? ????????? trajectory plot
        buffer = np.array(self.buffer)
        
        # set up figure and animation
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-self.env_boundary, self.env_boundary), ylim=(-self.env_boundary, self.env_boundary))
        ax.grid()

        robot, = ax.plot([], [], 'g', lw=2)
        link1, = ax.plot([], [], 'ko-', lw=2)
        link2, = ax.plot([], [], 'k', lw=2)
        gripper, = ax.plot([], [], 'k', lw=1)
        target, = ax.plot([], [], 'bo', ms=6)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        reward_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        robot_geom = np.array(
            [
                [0.3, -0.2, -0.2, 0.3],
                [  0,  0.2, -0.2,   0],
                [  1,    1,    1,   1]
            ]
        )
        link2_geom = np.array(
            [
                [-self.link2_len, -0.1],
                [              0,    0],
                [              1,    1]
            ]
        )
        gripper_geom = np.array(
            [
                [0.1, -0.1, -0.1,  0.1],
                [0.1,  0.1, -0.1, -0.1],
                [   1,    1,   1,    1]
            ]
        )

        def init():
            """initialize animation"""
            robot.set_data([], [])
            link1.set_data([], [])
            link2.set_data([], [])
            gripper.set_data([], [])
            target.set_data([], [])
            time_text.set_text('')
            reward_text.set_text('')
            return robot, link1, link2, gripper, target, time_text, reward_text

        def animate(i):
            """perform animation step"""
            robot_points = buffer[i]['robot'] * robot_geom
            link2_points = buffer[i]['link2'] * link2_geom
            gripper_points = buffer[i]['link2'] * gripper_geom

            robot.set_data((robot_points[0, :], robot_points[1, :]))
            link1.set_data((
                [buffer[i]['robot'].x(), buffer[i]['link1'].x()],
                [buffer[i]['robot'].y(), buffer[i]['link1'].y()]
            ))
            link2.set_data((link2_points[0, :], link2_points[1, :]))
            gripper.set_data((gripper_points[0, :], gripper_points[1, :]))
            target.set_data([buffer[i]['target'].x(), buffer[i]['target'].y()])
            time_text.set_text('time = %.1f' % buffer[i]['time'])
            reward_text.set_text('reward = %.3f' % buffer[i]['reward'])
            return robot, link1, link2, gripper, target, time_text, reward_text

        interval = self.dt * 1000
        ani = animation.FuncAnimation(fig, animate, frames=len(self.buffer),
                                        interval=interval, blit=True, init_func=init)

        plt.show()



def test(env):
    '''
    Test script for the environment "Manipulator2D"
    '''

    # ?????? ?????????
    env.reset()

    # 20??? ????????? ???????????? ??????
    for t in np.arange(0, 20, env.dt):
        # ??????????????? ?????? ????????? ????????? ?????? ???????????? ?????? ????????? ?????????????????? ????????? ??????

        # position error??? ????????? control input ??????
        link2_to_target = env.link2_tf_global.inv() * env.target_tf.get_translation()
        err1 = env.link2_tf * link2_to_target
        err2 = env.link1_tf * env.joint2_tf * err1
        err3 = env.joint1_tf * err2
        action = [
            np.linalg.norm(err3),
            np.arctan2(err3[1], err3[0]),
            np.arctan2(err2[1], err2[0]),
            np.arctan2(err1[1], err1[0])
        ]

        # Environment??? step ????????? ????????????, 
        # ????????? state(observation)??? reward, episode ????????????, ?????? ????????? ?????????
        next_state, reward, done, info = env.step(action)

        # episode ??????
        if done:
            break

    # Episode ????????? ????????? trajectory plot
    env.render()


if __name__=='__main__':

    test(Manipulator2D(tol=0.01))