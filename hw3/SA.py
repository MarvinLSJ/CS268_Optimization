import numpy as np
from math import exp, sin
import matplotlib.pyplot as plt

"""
Main Function Implementing Simulated Annealing
"""
def simulated_annealing(func, x0, allowed, t0, t_final, G=None, N_T=5, N_c=2, T_step=0.05, step = 0.9, verbose=False):
    if G is not None:
        # Execute graph coloring problem
        if verbose:
            print("Solving Graph 2-Coloring Problem")

        def wrapper_func(x, G=G):
            return func(x, G)

        def update_state(x, allowed, s):
            return graph_step(x, allowed)
    else:
        # Execute function optimization problem
        if verbose:
            print("Solving Function Optimization Problem")

        def wrapper_func(x):
            return func(x)

        def update_state(x, allowed, s):
            return func_step(x, allowed, s)

    curr_temp = t0
    ending_temp = t_final
    curr_state = x0
    curr_score = wrapper_func(curr_state)
    best = x0
    best_score = curr_score
    dim = len(curr_state)
    ac_ratio = np.ones(dim)

    while curr_temp > ending_temp:

        # Sweep
        N_loop1 = N_T
        while N_loop1 > 0:

            # Cycle
            N_loop2 = N_c
            while N_loop2 > 0:
                next_state, i = update_state(curr_state.copy(), allowed, step)

                # Boltzman
                delta_func = wrapper_func(next_state) - wrapper_func(curr_state)
                if delta_func < 0:
                    # if next is smaller, accept the change
                    curr_state = next_state
                else:
                    # if next is larger, accept with some probability
                    prob = exp(-delta_func / curr_temp)
                    threshold_prob = np.random.uniform()
                    if prob > threshold_prob:
                        # accept the change
                        curr_state = next_state
                    else:
                        # refuse
                        curr_state = curr_state
                        ac_ratio[i] = (ac_ratio[i] - 1 / N_c) if (ac_ratio[i] - 1 / N_c) > 0 else 1 / (1 + N_c)

                N_loop2 -= 1

            # update stepsize
            step = stepsize_factor(ac_ratio[i]) * step
            #             print(stepsize_factor(ac_ratio[i]))
            N_loop1 -= 1

        curr_temp = curr_temp - T_step

    best = curr_state
    best_score = wrapper_func(curr_state)

#     print(best, best_score)
    return best, best_score


# Helper function used to construct a uni-connected graph
def construct_graph(N):
    g = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if abs(i-j) == 1:
                if j > i and j%N != 0:
                    g[i][j] = 1
                elif i > j and i%N !=0:
                    g[i][j] = 1
            if abs(i-j) == N:
                g[i][j] = 1
    return g


# Flip elements in graph
def graph_step(x, states):
    # flip r-th element
    r = np.random.randint(len(x))
    # substitute randomly with another different value
    c = np.random.randint(len(states))
    while (states[c] == x[r]):
        c = np.random.randint(len(states))

    x[r] = states[c]
    return x, r


# Update real number value
def func_step(x, allowed, step, global_min=-100, global_max=100):
    dim = len(x)
    r = np.random.rand() * 2 - 1
    i = np.random.randint(dim)
    x[i] += 1.0 * r * step

    # If we have specified bound as left bound=allowed[0], right bound=allowed[1]
    if len(allowed) != 0:
        if x[i] < allowed[0] or x[i] > allowed[1]:
            return [(allowed[1] - allowed[0]) * np.random.random() + allowed[0]], i
    # If exceeding global bound, get a random number
    if x[i] < global_min or x[i] > global_max:
        return [200 * np.random.random() - 100], i

    return x, i


# Helper function g(accept_ratio) to adjust step size
def stepsize_factor(a, constant=2):
    c = constant
    if(a<0.4):
        return 1/(1+((c*(0.4-a))/0.4))
    elif(a>0.6):
        return 1 + (c*(a-0.6))/0.4
    else:
        return a


# Graph coloring problem evaluation function
def Graph_Coloring_Func(x, G):
    dim = len(x)
#     if(len(states) == 2):
#         return (np.trace(np.dot(G,np.outer(x,x.T))) + np.trace(np.dot(G,np.outer(1-x,(1-x).T))))/2
#     else:
    Esum = 0
    for i in range(dim):
        for j in range(dim):
            if x[i]==x[j] and i!=j and G[i][j] !=0:
                Esum += 1
    return Esum*1.0/2

if __name__=='__main__':

    # graph coloring test

    N = 30
    G = construct_graph(N)
    x_initial = np.zeros(N)
    t0 = 10
    t_final = 0.1
    allowed = [0,1,2]

    print(simulated_annealing(Graph_Coloring_Func, x_initial, allowed, t0, t_final, G))



    # Function optimization

    def func_test(x):
        return x[0] * sin((1 / 2) * x[0])

    r = 50
    func = func_test
    x_initial = [0.0]
    t0 = 10
    t_final = 0.1
    allowed = [-r, r]
    bx, bs = simulated_annealing(func_test, x_initial, allowed, t0, t_final)

    xs = np.linspace(-r, r, 1000)
    ys = [func([x]) for x in xs]
    plt.plot(xs, ys)
    plt.plot(x_initial, func(x_initial), 'g.', label='initial point')
    plt.plot(bx, bs, 'r.', label='SA result point')
    plt.legend()
    plt.show()
