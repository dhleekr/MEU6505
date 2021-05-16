from os import cpu_count
from numpy import (
    empty, linspace, argmax,
    max as npmax
)
from numpy.linalg import norm
from multiprocessing import Process, Queue



class ValueIteration(object):
    """
    ValueIteration(num_states, num_actions, rewards, state_transition_probs, discount)

    Finds an optimal value and a policy of a Markov decision process.
    Parameters
    ----------
    MDP : dict
        Dictionary of Markov decision process.
    MDP["num_states"] : int
        Number of elements in the set of states.
    MDP["num_actions'] : int
        Number of elements in the set of actions.
    MDP["rewards"] : numpy.ndarray
        Reward values in given states and actions.
        $r(s, a)$.
    MDP["state_transition_probs"] : numpy.ndarray
        Probability in transion to a next state $s'$ given state $s$ and action $a$.
    MDP["discount"] : float
        Discount factor, bounded by [0, 1]
    """

    def __init__(self, MDP):

        self.num_states = MDP["num_states"]
        self.num_actions = MDP["num_actions"]
        self.rewards = MDP["rewards"]
        self.discount = MDP["discount"]
        self.state_transition_probs = MDP["state_transition_probs"]
        self.policy = argmax(self.rewards, axis=1)

        # Initialize the current value estimate with zeros
        self.values = npmax(self.rewards, axis=1)
    

    def update(self):
        
        # Compute the action values $Q(s,a)$
        indices = linspace(0, self.num_states, cpu_count()+1).astype(int)
        values = empty((self.num_states,), dtype=self.rewards.dtype)
        def worker(queue, start, end):
            _values = empty((end-start,), dtype=self.rewards.dtype)
            _policy = empty((end-start,), dtype=self.policy.dtype)
            for s in range(start, end):
                _q = self.rewards[s, :] + self.discount * self.state_transition_probs[s] * self.values
                # Evaluate the deterministic policy $\pi(s)$
                _policy[s-start] = argmax(_q)
                # Compute the values $V(s)$
                _values[s-start] = npmax(_q)
            queue.put((_values, _policy))
        workers = list()
        for start, end in zip(indices[:-1], indices[1:]):
            queue = Queue()
            workers.append(
                (
                    start,
                    end,
                    queue,
                    Process(
                        target=worker,
                        args=(queue, start, end)
                    )
                )
            )
            workers[-1][-1].start()
        for (start, end, queue, process) in workers:
            _values_tmp, _policy_tmp = queue.get()
            values[start:end] = _values_tmp
            self.policy[start:end] = _policy_tmp
            process.join()

        # Compute the value difference $|\V_{k}-V_{k+1}|\$ for check the convergence
        diff = norm(
            (self.values[:] - values[:]) / self.num_states
        )

        # Update the current value estimate
        self.values = values

        return diff


    def solve(self, max_iteration=1e3, tolerance=1e-3, verbose=False, logging=False):
        
        if logging:
            history=[]

        # Value iteration loop
        for _iter in range(1, int(max_iteration+1)):

            # Update the value estimate
            diff = self.update()
            if logging:
                history.append(diff)
            if verbose:
                print('Iteration: {0}\tValue difference: {1}'.format(_iter, diff))

            # Check the convergence
            if diff < tolerance:
                if verbose:
                    print('Converged at iteration {0}.'.format(_iter))
                break

        sol = dict(
            values = self.values.copy(),
            policy = self.policy.copy()
        )

        if logging:
            return sol, history
        else:
            return sol



def test():
    """
    Test code for debugging
    """

    from scipy.sparse import dok_matrix, vstack
    from numpy import float32, zeros, ones
    from numpy.random import randint

    # Setup the number of states and actions
    n_states = 11
    n_actions = 3

    # Setup the reward $r(s,a)$
    rewards = zeros([n_states, n_actions])
    rewards[-1, -1] = 1

    # Setup the random state transition probability $P(s'|s,a)$
    p = float32(0.5)
    state_transition_probs = empty((n_states,), dtype=object)
    for i in range(n_states):
        P = dok_matrix(
            (1.0 - p) * ones((n_actions, n_states), dtype=float32) / float32(n_states - 1)
        )
        for j in range(n_actions):
            P[j, randint(low=0, high=n_states)] = p
        state_transition_probs[i] = P.tocsr()

    MDP = dict(
        n_states = n_states,
        n_actions = n_actions,
        rewards = rewards,
        state_transition_probs = state_transition_probs,
        discount = 0.99
    )

    # Test model
    solver = ValueIteration(MDP)
    solver.solve(
        max_iteration = 100,
        tolerance = 0.001,
        verbose = True,
        logging = False
    )



if __name__=='__main__':

    test()
