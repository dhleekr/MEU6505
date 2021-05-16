from os import cpu_count
from time import time
from numpy import (
    eye, linspace, empty, argmax,
    max as npmax
)
from numpy.linalg import norm
from scipy.sparse import identity, vstack
from scipy.sparse.linalg import spsolve
from multiprocessing import Process, Queue



class PolicyIteration(object):
    """
    PolicyIteration(num_states, num_actions, rewards, state_transition_probs, discount)

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

        # Identity matrix $I_{|s|}$ and $I_{|a|}$ for computation
        self._I_s = identity(self.num_states, dtype=self.rewards.dtype, format='csr')
        self._I_a = eye(self.num_actions)

        # Initialize the value estimate with $\infty$ for compute the value difference at first iteration
        self.values = npmax(self.rewards, axis=1)

        # Initilaize the random determinist policy
        self.policy = argmax(self.rewards, axis=1)
        self.dense = False


    def update(self):
        
        # Compute the value $V(s)$ via solving the linear system $(I-\gamma P^{\pi}), R^{\pi}$
        _A = self._I_s - self.discount * vstack(
            [self.state_transition_probs[s].getrow(a) for s, a in enumerate(self.policy)],
            format='csr'
        )
        _b = self.rewards[self._I_a[self.policy].astype(bool)]
        values = spsolve(_A, _b)    # Solve linear system _Ax = _b 

        # Update the deterministic policy $\pi(s)$
        indices = linspace(0, self.num_states, cpu_count()+1).astype(int)
        def worker(queue, start, end):
            _policy = empty((end-start,), dtype=self.policy.dtype)
            for s in range(start, end):
                _policy[s-start] = argmax(
                    self.rewards[s, :] + self.discount * self.state_transition_probs[s] * values
                )
            queue.put(_policy)
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
            self.policy[start:end] = queue.get()
            process.join()

        # Compute the value difference $|\V_{k}-V_{k+1}|\$ for check the convergence
        diff = norm(
            (self.values[:] - values[:]) / self.num_states
        )

        # Update the current value estimate
        self.values = values.copy()

        return diff


    def solve(self, max_iteration=1e3, tolerance=1e-3, verbose=False, logging=False):
        
        if logging:
            history=[]

        # Policy iteration loop
        for _iter in range(1, int(max_iteration+1)):

            # Update the value estimate and the polcy estimate
            _startTime = time()
            diff = self.update()
            _endTime = time()
            if logging:
                history.append(diff)
            if verbose:
                print(
                    'Iteration: {0}\tValue difference: {1}\tTime: {2}'.format(
                        _iter, diff, _endTime - _startTime
                    )
                )

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

    from scipy.sparse import dok_matrix
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
    state_transition_probs = empty((n_states), dtype=object)
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
    solver = PolicyIteration(MDP)
    solver.solve(
        max_iteration = 100,
        tolerance = 0.001,
        verbose = True,
        logging = False
    )



if __name__=='__main__':

    test()
