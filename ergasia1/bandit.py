import numpy as np
import random

class BanditWorld:
    def __init__(self, machines, levers, std_dev):
        self.machines = machines
        self.levers = levers  
        self.std  = std_dev
        
        self.true_means = np.random.randint(0, 1001, size = (machines, levers))
        
    def print_true_means(self):
        print("Means", end="\n")
        print("      ", end="")
        for l in range(self.levers):
            print(f"L{l:<6}", end="")
        print()

        for m in range(self.machines):
            print(f"M{m:<3}  ", end="")
            for l in range(self.levers):
                print(f"{self.true_means[m][l]:<6}", end=" ")
            print()
        print()

        
    def give_reward(self, machine, lever):
        if machine > self.machines - 1:
            exit("[ERROR] Machine out of bounds")

        if  lever > self.levers - 1:
            exit("[ERROR] Lever out of bounds")

        true_mean = float(self.true_means[machine][lever])
        reward = round(random.gauss(true_mean, float(self.std)), 2)
        
        if reward <= 0:
            reward = float(0)
            
        return reward        