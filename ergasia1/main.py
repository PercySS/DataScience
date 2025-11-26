import sys
import bandit
import agent
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sns

def plot_cum_per_episode(episodes_range, epsilon_rewards, softmax_rewards):
    plt.figure(figsize=(12,6))
    plt.plot(episodes_range, epsilon_rewards, label="Epsilon-Greedy", linewidth=2)
    plt.plot(episodes_range, softmax_rewards, label="Softmax", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward per Episode")
    plt.legend()
    plt.show()
    
def plot_rew_per_ation_avg_episode(e_agents, soft_agents):
    min_actions = min(
                        min(len(a.reward_per_step) for a in e_agents),
                        min(len(a.reward_per_step) for a in soft_agents)
                        )

    # fine tuning for plots i cut the lists in order to have same length
    avg_reward_per_action_e = np.mean([a.reward_per_step[:min_actions] for a in e_agents], axis=0)
    avg_reward_per_action_s = np.mean([a.reward_per_step[:min_actions] for a in soft_agents], axis=0)

    # plot
    plt.figure(figsize=(12,6))
    plt.plot(np.arange(min_actions), avg_reward_per_action_e, label="Epsilon-Greedy")
    plt.plot(np.arange(min_actions), avg_reward_per_action_s, label="Softmax")
    plt.xlabel("Action")
    plt.ylabel("Average Reward")
    plt.title("Reward History per Algorithm (averaged over episodes)")
    plt.legend()
    plt.show()
    
def plot_violins(epsilon_rewards, softmax_rewards):
    plt.figure(figsize=(8,6))
    sns.violinplot(data=[epsilon_rewards, softmax_rewards])
    plt.xticks([0,1], ["Epsilon-Greedy", "Softmax"])
    plt.ylabel("Cumulative Reward")
    plt.title("Distribution of Cumulative Rewards")
    plt.show()
    
def plot_bars(epsilon_rewards, softmax_rewards):
    mean_rewards = [np.mean(epsilon_rewards), np.mean(softmax_rewards)]
    plt.figure(figsize=(6,5))
    plt.bar(["Epsilon-Greedy", "Softmax"], mean_rewards, color=["skyblue", "salmon"])
    plt.ylabel("Mean Cumulative Reward")
    plt.title("Mean Performance of Agents")
    plt.show()

def plot_superchart(e_agents, soft_agents):
    # we must cut to the least min of the episodes (fine tuning for plot)
    min_actions =   min(
                        min(len(a.reward_per_step) for a in e_agents),
                        min(len(a.reward_per_step) for a in soft_agents)
                    )

    # transfer to array
    e_rewards_matrix = np.array([a.reward_per_step[:min_actions] for a in e_agents])
    s_rewards_matrix = np.array([a.reward_per_step[:min_actions] for a in soft_agents])

    # means and std
    avg_e = np.mean(e_rewards_matrix, axis=0)
    std_e = np.std(e_rewards_matrix, axis=0)

    avg_s = np.mean(s_rewards_matrix, axis=0)
    std_s = np.std(s_rewards_matrix, axis=0)

    plt.figure(figsize=(12,6))
    plt.plot(np.arange(min_actions), avg_e, label="Epsilon-Greedy", color="blue")
    plt.fill_between(np.arange(min_actions), avg_e - std_e, avg_e + std_e, color="blue", alpha=0.2)

    plt.plot(np.arange(min_actions), avg_s, label="Softmax", color="red")
    plt.fill_between(np.arange(min_actions), avg_s - std_s, avg_s + std_s, color="red", alpha=0.2)

    plt.xlabel("Action")
    plt.ylabel("Average Reward ± Std")
    plt.title("Reward Evolution per Action with Variance")
    plt.legend()
    plt.show()
    


def main():

    epsilon = 0.1
    tau = 0.6
    
    if (len(sys.argv) < 6): 
        exit("[USAGE] python3 <num_machines> <num_levers_per_m> <num_actions> <std_deviation> <num_episodes>")
        
    
    try:
        machines = int(sys.argv[1]) if int(sys.argv[1]) > 0 else exit("Machines must be > 0.")
        levers = int(sys.argv[2]) if int(sys.argv[2]) > 0 else exit("Levers per machine must be > 0.")
        actions = int(sys.argv[3]) if int(sys.argv[3]) > 1 else exit("Total actions must be > 1 in order to make sense.")
        std_dev = int(sys.argv[4]) if int(sys.argv[4]) > 0 else exit("Standard deviation must be > 0 in order to not be boring.")
        episodes = int(sys.argv[5]) if int(sys.argv[4]) > 1 else exit("Number of episodes must be > 1 in order to see difference overtime.")
    except (ValueError, IndexError):
        exit(f"[Error] Not provided an int.")
    
    e_agents = []
    soft_agents = []
    
    epsilon_rewards = []
    softmax_rewards = []
    
    epsilon_histories = []
    softmax_histories = []
    
    for _ in range(episodes):
        # i make a world
        world = bandit.BanditWorld(machines, levers, std_dev)
    
        # i make two agents with different algorithms
        agentEpsilon = agent.Agent(actions, machines, levers, epsilon, -1)
        agentSoftmax = agent.Agent(actions, machines, levers, -1, tau)
        
        # let those dudes gamble their life savings and their kids' college tuition away with different algorithms
        agentEpsilon.run(world)
 
        agentSoftmax.run(world)

         
        # i put them in lists in order to make later the plots
        e_agents.append(agentEpsilon)
        soft_agents.append(agentSoftmax)   
        
        epsilon_rewards.append(agentEpsilon.cum_reward)
        softmax_rewards.append(agentSoftmax.cum_reward)
        
        epsilon_histories.append(agentEpsilon.choice_per_step)
        softmax_histories.append(agentSoftmax.choice_per_step)
        
        del agentEpsilon
        del agentSoftmax
        del world
        
    
    episodes_range = np.arange(1, len(e_agents)+1)
            
    
    ##################################################
    #               mini GUI for plots               #
    ##################################################
    
    def show_plots():
        if var1.get():
            plot_cum_per_episode(episodes_range, epsilon_rewards, softmax_rewards)
        if var2.get():
            plot_rew_per_ation_avg_episode(e_agents, soft_agents)
        if var3.get():
            plot_violins(epsilon_rewards, softmax_rewards)
        if var4.get():
            plot_bars(epsilon_rewards, softmax_rewards)
        if var5.get():
            plot_superchart(e_agents, soft_agents)
    
    root = tk.Tk()
    root.title("Choose diagrams")
    root.geometry('280x170+800+400')

    var1 = tk.BooleanVar()
    var2 = tk.BooleanVar()
    var3 = tk.BooleanVar()
    var4 = tk.BooleanVar()
    var5 = tk.BooleanVar()
    

    tk.Checkbutton(root, text="Cumulative Reward per Episode", variable=var1).pack()
    tk.Checkbutton(root, text="Reward History per Algorithm", variable=var2).pack()
    tk.Checkbutton(root, text="Distribution of Cumulative Rewards", variable=var3).pack()
    tk.Checkbutton(root, text="Mean Performance of Agents", variable=var4).pack()
    tk.Checkbutton(root, text="Super Chart", variable=var5).pack()

    tk.Button(root, text="Show Selected Plots", command=show_plots).pack()

    root.mainloop()

    

    
if __name__ == "__main__":
    main()