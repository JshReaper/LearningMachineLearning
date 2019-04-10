import numpy as np
import sys
from gridworld import GridworldEnv


gamma =1.0
env = GridworldEnv()


# it calculates the value function given a policy
def policy_evaluation(policy):
    V = np.zeros(env.nS)  # intialize V to 0's

    while True:
        delta = 0

        for s in range(env.nS):  # runs for 16 states  (0-15)
            total_state_value = 0

            for a, prob_a in enumerate(policy[s]):  # runs for 4 actions (0-3)

                for prob_s, next_state, reward, _ in env.P[s][a]:
                    total_state_value += prob_a * prob_s * (reward + gamma * V[next_state])

            delta = max(delta, np.abs(total_state_value - V[s]))  # calculate change
            V[s] = total_state_value

        # a check to terminate
        if delta < 0.005:
            break

    return np.array(V)


# it improves the policy given a value function
def policy_improvement(V, policy):
    #     policy = np.ones([env.nS, env.nA]) / env.nA

    for s in range(env.nS):

        Q_sa = np.zeros(env.nA)

        for a in range(env.nA):
            for prob_s, next_state, reward, _ in env.P[s][a]:
                Q_sa[a] += prob_s * (reward + gamma * V[next_state])

        best_action = np.argmax(Q_sa)

        policy[s] = np.eye(env.nA)[best_action]

    return policy


def policy_iteration():
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    epochs = 1000
    for i in range(epochs):

        V = policy_evaluation(policy)

        old_policy = np.copy(policy)

        new_policy = policy_improvement(V, old_policy)

        if (np.all(policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        policy = new_policy

    return policy, V


final_policy, final_v = policy_iteration()


print("Final Policy ")
print(final_policy)

print("Final Policy grid : (0=up, 1=right, 2=down, 3=left)")
print(np.reshape(np.argmax(final_policy, axis=1), env.shape))

print("Final Value Function grid")
print(final_v.reshape(env.shape))


def optimal_value_function(V):
    while True:
        delta = 0
        for s in range(env.nS):

            Q_sa = np.zeros(env.nA)

            for a in range(env.nA):
                for prob_s, next_state, reward, _ in env.P[s][a]:
                    Q_sa[a] += prob_s * (reward + gamma * V[next_state])

            max_value_function_s = np.max(Q_sa)

            delta = max(delta, np.abs(max_value_function_s - V[s]))

            V[s] = max_value_function_s

        if delta < 0.00001:
            break

    return V


def optimal_policy_extraction(V):
    policy = np.zeros([env.nS, env.nA])

    for s in range(env.nS):
        Q_sa = np.zeros(env.nA)

        for a in range(env.nA):
            for prob_s, next_state, reward, _ in env.P[s][a]:
                Q_sa[a] += prob_s * (reward + gamma * V[next_state])

        best_action = np.argmax(Q_sa)

        policy[s] = np.eye(env.nA)[best_action]

    return policy


def value_iteration():
    V = np.zeros(env.nS)

    optimal_v = optimal_value_function(V)

    policy = optimal_policy_extraction(optimal_v)

    return policy, optimal_v


final_policy,final_v = value_iteration()


print("Final Policy ")
print(final_policy)

print("Final Policy grid : (0=up, 1=right, 2=down, 3=left)")
print(np.reshape(np.argmax(final_policy, axis=1), env.shape))

print("Final Value Function grid")
print(final_v.reshape(env.shape))
