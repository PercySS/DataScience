import numpy as np
import matplotlib.pyplot as plt

gamma = 1.0
reward = -1

actions = {
    "up":    -4,
    "down":   4,
    "left":  -1,
    "right":  1
}

terminal_states = [0, 15]
states = [i for i in range(16) if i not in terminal_states]


def next_state(s, action):
    ns = s + actions[action]

    # bounds of grid
    if ns < 0 or ns > 15:
        return s
    if s % 4 == 0 and action == "left":
        return s
    if s % 4 == 3 and action == "right":
        return s

    return ns


def policy_evaluation_two_arrays(theta=1e-4):
    V = np.zeros(16)

    while True:
        delta = 0
        V_new = V.copy()

        for s in states:
            value = 0
            for a in actions:
                ns = next_state(s, a)
                value += 0.25 * (reward + gamma * V[ns])

            V_new[s] = value
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < theta:
            break

    return V


def policy_evaluation_one_array(theta=1e-4):
    V = np.zeros(16)

    while True:
        delta = 0
        for s in states:
            old_v = V[s]
            value = 0

            for a in actions:
                ns = next_state(s, a)
                value += 0.25 * (reward + gamma * V[ns])

            V[s] = value
            delta = max(delta, abs(old_v - V[s]))

        if delta < theta:
            break

    return V


V_two = policy_evaluation_two_arrays()
V_one = policy_evaluation_one_array()


def print_values(V):
    grid = V.reshape(4,4)
    grid[0,0] = 0
    grid[3,3] = 0
    print(np.round(grid, 2))

print_values(V_two)



def optimal_actions(V):
    policy = {}

    for s in states:
        best_value = -1e9
        best_actions = []

        for a in actions:
            ns = next_state(s, a)
            val = reward + gamma * V[ns]

            if val > best_value:
                best_value = val
                best_actions = [a]
            elif val == best_value:
                best_actions.append(a)

        policy[s] = best_actions

    return policy

opt_policy = optimal_actions(V_two)

def plot_values(V):
    grid = V.reshape(4,4)
    grid[0,0] = 0
    grid[3,3] = 0

    plt.imshow(grid, cmap="coolwarm")
    plt.colorbar(label="V(s)")
    plt.title("State Value Function")
    plt.show()

def plot_grid_with_policy(V, policy):
    fig, ax = plt.subplots(figsize=(6, 6))

    grid = V.reshape(4, 4)
    grid[0, 0] = 0
    grid[3, 3] = 0

    ax.imshow(grid, cmap="coolwarm")

    # draw grid lines
    for i in range(5):
        ax.axhline(i - 0.5, color='black')
        ax.axvline(i - 0.5, color='black')

    arrow_map = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→"
    }

    for s in range(16):
        if s in [0, 15]:
            continue

        row = s // 4
        col = s % 4

        actions = policy[s]
        arrows = "".join([arrow_map[a] for a in actions])

        ax.text(col, row, arrows,
                ha='center', va='center',
                fontsize=18, color='black')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Gridworld: Values + Optimal Actions")
    plt.show()


plot_values(V_two)

plot_grid_with_policy(V_two, opt_policy)

