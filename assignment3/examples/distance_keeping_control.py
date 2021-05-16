from os import cpu_count
from numpy import (
    float32, finfo, empty, zeros, linspace, arange, pi, cos, sin
)
from scipy.sparse import dok_matrix
from multiprocessing import Process, Queue



def constructMDP(sigma=0):

    r_min = 10
    r_max = 90
    delta_r = 0.4

    alpha_max = pi
    alpha_min = - alpha_max
    delta_alpha = 2 * pi / 361

    beta = 1

    keeping_dist = 50

    stochastic_term_1 = sigma ** 2 / delta_r ** 2 / 2   # for numerical calculation HJB eq

    r_list = linspace(r_min, r_max, int(r_max / delta_r) + 1)
    alpha_list = arange(alpha_min, alpha_max, delta_alpha)
    n_r = len(r_list)
    n_alpha = len(alpha_list)

    n_states = n_r * n_alpha

    action_list = [-1.0, 1.0]
    n_actions = len(action_list)

    state_transition_probs = empty((n_states,), dtype=object)
    rewards = zeros((n_states, n_actions), dtype=float32)
    _indices = linspace(0, n_r, cpu_count()+1).astype(int)

    def worker(queue, start, end):
        _probs_tmp = empty(((end-start)*n_alpha,), dtype=object)
        _rewards_tmp = zeros(((end-start)*n_alpha, n_actions), dtype=float32)
        for i, r in zip(range(start, end), r_list[start:end]):
            for j, alpha in enumerate(alpha_list):  # 2nd order partial differential equation of HJB eq
                P = dok_matrix((n_actions, n_states), dtype=float32)
                stochastic_term_2 = sigma ** 2 / delta_alpha ** 2 / r ** 2 / 4
                b1 = -cos(alpha) + sigma**2 / (2*r)**2
                s = i * n_alpha + j
                for a, u in enumerate(action_list):
                    b2 = sin(alpha) / r - u
                    p1_p = max(0, b1) / delta_r + stochastic_term_1  # \frac{b_1^+}{\Delta r} + \frac{\sigma_T^2}{2\Delta r^2}
                    p1_m = max(0, -b1) /delta_r + stochastic_term_1  # \frac{b_1^-}{\Delta r} + \frac{\sigma_T^2}{2\Delta r^2}
                    p2_p = max(0, b2) / delta_alpha + stochastic_term_2  # \frac{b_2^+}{\Delta\alpha} + \frac{\sigma_T^2}{4r^2\Delta\alpha^2}
                    p2_m = max(0, -b2) / delta_alpha + stochastic_term_2  # \frac{b_2^-}{\Delta\alpha} + \frac{\sigma_T^2}{4r^2\Delta\alpha^2}
                    p_denominator = p1_m + p1_p + p2_m + p2_p + beta
                    if i == n_r-1:
                        P[a, s] += p1_p / p_denominator
                    else:
                        P[a, s + n_alpha] += p1_p / p_denominator
                    if i == 0:
                        P[a, s] += p1_m / p_denominator
                    else:
                        P[a, s - n_alpha] += p1_m / p_denominator
                    if j == n_alpha-1:
                        P[a, s - j] += p2_p /p_denominator
                    else:
                        P[a, s + 1] += p2_p /p_denominator
                    if j == 0:
                        P[a, s + n_alpha - 1] += p2_m / p_denominator
                    else:
                        P[a, s - 1] += p2_m / p_denominator
                    _rewards_tmp[s-start*n_alpha, a] -= (r - keeping_dist) ** 2 / p_denominator
                _probs_tmp[s-start*n_alpha] = P.tocsr()
        queue.put((_probs_tmp, _rewards_tmp))
    workers = list()
    for start, end in zip(_indices[:-1], _indices[1:]):
        queue = Queue()
        workers.append(
            (
                queue,
                start,
                end,
                Process(target=worker, args=(queue, start, end))
            )
        )
        workers[-1][-1].start()
    for queue, start, end, p in workers:
        _probs_tmp, _rewards_tmp = queue.get()
        state_transition_probs[start*n_alpha:end*n_alpha] = _probs_tmp
        rewards[start*n_alpha:end*n_alpha] = _rewards_tmp
        p.join()

    return dict(
        num_states = n_states,
        num_actions = n_actions,
        rewards = rewards,
        state_transition_probs = state_transition_probs,
        discount = 1-finfo(float32).eps
    )
