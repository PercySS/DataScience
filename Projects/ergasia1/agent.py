import numpy as np
import random

class Agent:
    def __init__(self, actions, machines, levers, epsilon, tau):
        self.machines = machines
        self.levers = levers
        self.actions = actions
        self.estimates = np.zeros((machines, levers))
        self.counts = np.zeros((machines, levers))
        self.cum_reward = 0
        self.reward_per_step = []
        self.choice_per_step = []
        self.epsilon = epsilon
        self.tau = tau
        
    def e_greedy(self):
        # i choose random float in [0, 1] (default construtor)
        r = random.random()

        # r < epsilon i explore
        # r >= epsilon i exploit
        if r < self.epsilon:
            # randomly select lever for exploration
            machine = random.randint(0, self.machines - 1)
            lever = random.randint(0, self.levers - 1)
        else: 
            # select highest reward estimate from my history for exploitation
            machine, lever = np.unravel_index(self.estimates.argmax(), self.estimates.shape)
        return machine, lever
        
    def softmax(self):
        max_est = np.max(self.estimates)

        exp_est = np.exp((self.estimates - max_est) / self.tau)
        
        # compute probabilities
        sum_exp_est = np.sum(exp_est)
        

        if sum_exp_est == 0:
            probs = np.ones_like(self.estimates) / self.estimates.size
        else:
            probs = exp_est / sum_exp_est

        flat_index = np.random.choice(probs.size, p=probs.flatten())
        
        # unravel the flat index back to (machine, lever)
        machine, lever = np.unravel_index(flat_index, self.estimates.shape)
        
        return machine, lever
        
    def choose_action(self):
        if self.epsilon == -1:
            machine, lever = self.softmax()
        else: 
            machine, lever = self.e_greedy()
        return machine, lever
            
    def update(self, machine, lever, reward):
        self.counts[machine][lever] += 1
        
        # estimates from average rewards 
        self.estimates[machine][lever] += (reward - self.estimates[machine][lever]) / self.counts[machine][lever]
        self.cum_reward += reward 
        self.reward_per_step.append(reward)
        self.choice_per_step.append((machine, lever))
        
        
    def run(self, world):
        if self.tau != -1:
            # initial exploration pass for softmax algorithm: play everything once
            for m in range(self.estimates.shape[0]):
                for l in range(self.estimates.shape[1]):
                    reward = world.give_reward(m, l)
                    self.update(m, l, reward)
        for i in range(self.actions):
            machine, lever = self.choose_action()

            reward = world.give_reward(machine, lever)
            
            self.update(machine, lever, reward)

            
            
    def print_estimations(self, machines, levers):
        print("Agent Epsilon-Greedy ", end ="\n") if self.epsilon != -1 else print("Agent SoftMax ", end ="\n")  
        print("      ", end="")
        for l in range(levers):
            print(f"L{l:<6}", end="")
        print()

        for m in range(machines):
            print(f"M{m:<3}  ", end="")
            for l in range(levers):
                print(f"{self.estimates[m][l]:<6.2f}", end=" ")
            print()
        print()


    