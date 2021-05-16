import numpy as np
import matplotlib.pyplot as plt
from grid_world import GridWorld
from settings import *

from time import time

state_dict = {(0,0): 0, (0,1): 1, (0,2): 2, (0,3): 3, \
             (1,0): 4, (1,1): 5, (1,2): 6, (1,3): 7, \
             (2,0): 8, (2,1): 9, (2,2): 10, (2,3): 11}
action_dict = {"up": 0, "down": 1, "left": 2, "right": 3}

state_keys = list(state_dict.keys())
state_vals = list(state_dict.values())
action_keys = list(action_dict.keys())
action_vals = list(action_dict.values())

class Agent:

    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.grid_world = GridWorld()
        self.isEnd = self.grid_world.isEnd
        self.lr = 0.1
        self.exp_rate = 0.3
        self.decay_gamma = 0.9
        self.game_len = []
        
        self.model =  Model(12, 4)
        self.n = 5 # model learning

        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = -np.inf
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.grid_world.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.grid_world.state, action))
        return action

    def takeAction(self, action):
        position = self.grid_world.nxtPosition(action)
        # update GridWorld
        return GridWorld(state=position)

    def reset(self):
        self.states = []
        self.grid_world = GridWorld()
        self.isEnd = self.grid_world.isEnd

    def play(self, episodes=100, verbose=False, logging=False):
        tic = time()
        done_epi = 0
        last_q_values_vec = np.zeros((3,4))
        convergence = 0
        
        if logging:
            history=[]
        
        for episode in range(episodes):
            self.reset()
            state = self.grid_world.state

            i = 0
            while True:
                i += 1
                # 현재 상태에 대한 행동 선택
                action = self.chooseAction()

                # 행동을 취한 후 다음 상태, 보상 받아옴
                self.grid_world = self.takeAction(action) 
                next_state = self.grid_world.state# next state
                reward = self.grid_world.giveReward() # reward

                # <s,a,r,s'>로 큐함수를 업데이트
                q_1 = self.Q_values[state][action] # current_q_value
                q_2 = reward + self.decay_gamma * max(self.Q_values[next_state].values())
                self.Q_values[state][action] += self.lr * (q_2 - q_1)

                tic_p = time()
                # Learn model
                self.model.add(state, action, next_state, reward)
                
                # Planning for n step
                self.planning()
                toc_p = time()
                
                state = next_state

                if state == (0, 3):
                    self.game_len.append(i)
                    break

            if (done_epi == 0):
                # Check the convergence
                q_values_vec = np.zeros((3,4))
                for i in range(3):
                    for j in range(4):
                        q_values_vec[i][j] = max(self.Q_values[(i,j)].values())

                diff = np.linalg.norm(q_values_vec[:] - last_q_values_vec[:], 2)
                
                if logging:
                    history.append(diff)
                if verbose:
                    print('Iteration: {0}\tValue difference: {1}'.format(episode, diff))
                    
                last_q_values_vec = q_values_vec

                # Check the convergence
                tolerance = 1e-2
                if diff < tolerance:
                    if verbose:
                        print('Converged at iteration {0}.'.format(episode))
                    done_epi = episode
                    self.done_epi = done_epi
                    toc = time()
                    convergence = (toc - tic) - (toc_p - tic_p)
        
        if logging:
            return history, convergence
        
    def planning(self):
        for i in range(self.n):
            state, action =  self.model.sample()
            next_state, reward = self.model.step(state,action)
            
            state = state_keys[state_vals.index(state)]
            next_state = state_keys[state_vals.index(next_state)]
            action = action_keys[action_vals.index(action)]
            
            
            q_1 = self.Q_values[state][action] # current_q_value
            q_2 = reward + self.decay_gamma * max(self.Q_values[next_state].values())
            self.Q_values[state][action] += self.lr * (q_2 - q_1)

    

class Model():
    def __init__(self, n_states, n_actions):
        self.transitions = np.zeros((n_states,n_actions), dtype=np.uint8)
        self.rewards = np.zeros((n_states, n_actions))

    def add(self,s,a,s_prime,r):
        s = state_dict[s]
        a = action_dict[a]
        s_prime = state_dict[s_prime]
        
        self.transitions[s,a] = s_prime
        self.rewards[s,a] = r

    def sample(self):
        """ Return random state, action"""
        # Random visited state
        s = np.random.choice(np.where(np.sum(self.transitions, axis=1) > 0)[0])
        # Random action in that state
        a = np.random.choice(np.where(self.transitions[s] > 0)[0])

        return s,a

    def step(self, s,a):
        """ Return state_prime and reward for state-action pair"""
       
        s_prime = self.transitions[s,a]
        r = self.rewards[s,a]
        return s_prime, r


if __name__ == "__main__":
    mc_sampling = 100
    episodes = 500
    game_len = [0 for i in range(episodes)]
    converge_time = []
    avg_time = []
    converge_iteration = []
    avg_iteration = []
    
    for i in range(mc_sampling):
        ag = Agent()
        _, converge = ag.play(episodes, logging=True)
        converge_time.append(converge)
        avg = sum(converge_time, 0) / len(converge_time)
        avg_time.append(avg)

        game_len = [a+b for a,b in zip(game_len, ag.game_len)]
        converge_iteration.append(ag.done_epi)
        del ag
        print("[MC Sampling]----(", i, "/", mc_sampling, ")")
    game_len = [x / mc_sampling for x in game_len]

    avg_steps = []
    for i in range(len(game_len)):
        steps = 0
        for j in range(i):
            steps += game_len[j]
        avg_steps.append(steps / (i+1))

    for i in range(len(converge_iteration)):
        iterations = 0
        for j in range(i):
            iterations += converge_iteration[j]
        avg_iteration.append(iterations / (i+1))

    print('Average steps :', avg_steps[-1])
    print('Average converge time :', avg_time[-1])
    print('Average converge iteration :',avg_iteration[-1])

    plt.figure(1)
    plt.plot(game_len, label = 'Dyna(model-based RL)')
    plt.plot(avg_steps, label = 'average steps')
    plt.title('Dyna(model-based RL)')
    plt.legend()
    plt.xlabel('episode')
    plt.ylabel('steps')
    plt.grid(True)

    plt.figure(2)
    plt.plot(converge_time, label = 'converge time')
    plt.plot(avg_time, label='average converge time')
    plt.title('Dyna(model-based RL) converge time')
    plt.xlabel('episode')
    plt.ylabel('time')
    plt.legend()
    plt.grid(True)
    
    plt.figure(3)
    plt.plot(converge_iteration, label='converge iteration')
    plt.plot(avg_iteration, label='average iteration')
    plt.title('Dyna(model-based RL) converge iteration')
    plt.xlabel('episode')
    plt.ylabel('iteration')
    plt.legend()
    plt.grid(True)
    plt.show()