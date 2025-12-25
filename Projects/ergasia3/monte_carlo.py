import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random


def draw_card():
    # we deal ccards from infinite deck
    card = random.randint(1, 13)
    return min(card, 10)  # figures and tens count as tens



def usable_ace(hand):
    # i see if there is ace that countys as 11 without busting
    return 1 in hand and sum(hand) + 10 <= 21


def hand_value(hand):
    value = sum(hand)
    if usable_ace(hand):
        value += 10
    return value


def is_bust(hand):
    return hand_value(hand) > 21


def dealer_policy(hand):
    # deal hits until over 17
    while hand_value(hand) < 17:
        hand.append(draw_card())
    return hand


def play_episode(policy, start_state=None, start_action=None):
    episode = []

    # Exploring start
    if start_state:
        player_sum, dealer_card, usable = start_state
        player_hand = [player_sum - 10 if usable else player_sum]
        if usable:
            player_hand.append(1)
        dealer_hand = [dealer_card, draw_card()]
    else:
        player_hand = [draw_card(), draw_card()]
        dealer_hand = [draw_card(), draw_card()]

    dealer_card = dealer_hand[0]

    # first action always random
    action = start_action

    while True:
        player_sum = hand_value(player_hand)
        usable = usable_ace(player_hand)

        if player_sum < 12:
            action = 1  # always hit
        else:
            state = (player_sum, dealer_card, usable)
            if action is None:
                action = policy[state]

        episode.append((state, action))

        if action == 1:  # hit
            player_hand.append(draw_card())
            if is_bust(player_hand):
                return episode, -1
        else:  # stick
            break

        action = None

    # dealer
    dealer_hand = dealer_policy(dealer_hand)

    player_total = hand_value(player_hand)
    dealer_total = hand_value(dealer_hand)

    if dealer_total > 21 or player_total > dealer_total:
        reward = 1
    elif player_total < dealer_total:
        reward = -1
    else:
        reward = 0

    return episode, reward



def monte_carlo_es(episodes):
    Q = defaultdict(float)
    returns = defaultdict(list)

    policy = defaultdict(int)

    # first policy only on 20 or 21 sticking 
    for ps in range(12, 22):
        for dc in range(1, 11):
            for ua in [True, False]:
                policy[(ps, dc, ua)] = 0 if ps >= 20 else 1

    for _ in range(episodes):
        ps = random.randint(12, 21)
        dc = random.randint(1, 10)
        ua = random.choice([True, False])

        start_state = (ps, dc, ua)
        start_action = random.choice([0, 1])

        episode, reward = play_episode(policy, start_state, start_action)
        visited = set()

        for state, action in episode:
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(reward)
                Q[(state, action)] = np.mean(returns[(state, action)])

                # policy improvement
                hit = Q[(state, 1)]
                stick = Q[(state, 0)]
                policy[state] = 1 if hit > stick else 0

    return policy, Q

def plot_policy(policy, usable):
    data = np.zeros((10, 10))

    for ps in range(12, 22):
        for dc in range(1, 11):
            data[21 - ps, dc - 1] = policy[(ps, dc, usable)]

    plt.imshow(data, cmap="coolwarm")
    plt.colorbar(label="0 = stick, 1 = hit")
    plt.xticks(range(10), range(1, 11))
    plt.yticks(range(10), range(21, 11, -1))
    plt.xlabel("Dealer card")
    plt.ylabel("Player sum")
    plt.title(f"Policy (usable ace = {usable})")
    plt.show()


def plot_policy_scatter(policy, usable):
    hit_x, hit_y = [], []
    stick_x, stick_y = [], []

    for ps in range(12, 22):
        for dc in range(1, 11):
            action = policy[(ps, dc, usable)]
            if action == 1:  # hit
                hit_x.append(dc)
                hit_y.append(ps)
            else:  # stick
                stick_x.append(dc)
                stick_y.append(ps)

    plt.figure(figsize=(8, 6))
    plt.scatter(hit_x, hit_y, marker='o', label='Hit', alpha=0.7)
    plt.scatter(stick_x, stick_y, marker='s', label='Stick', alpha=0.7)

    plt.xticks(range(1, 11))
    plt.yticks(range(12, 22))
    plt.xlabel("Dealer card")
    plt.ylabel("Player sum")
    plt.title(f"Policy decision plot (usable ace = {usable})")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    policy, Q = monte_carlo_es(500000)
    print("Training finished")

    plot_policy(policy, usable=True)
    plot_policy(policy, usable=False)

    plot_policy_scatter(policy, usable=True)
    plot_policy_scatter(policy, usable=False)


if __name__ == "__main__":
    main()
