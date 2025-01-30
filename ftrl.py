import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm


class FTRL:
    """
    FTRL
    """
    # def __init__(self, env, episodes, alpha, beta, gamma = 1):
    def __init__(self, env, episodes):
        self.env = env
        self.episodes = episodes
        # self.alpha = alpha
        # self.beta = beta
        # self.gamma = gamma
        self.counts = np.zeros((self.env.ns, self.env.na))
        self.loss_history = np.zeros((self.episodes))

    def reset(self):
        self.counts = np.zeros((self.env.ns, self.env.na))
        self.loss_history = np.zeros((self.episodes))


    def execute(self):
        cumulative_loss = np.zeros((self.env.ns, self.env.na))  # Initialize cumulative loss for each action

        for t in tqdm(range(self.episodes)):
            s = self.env.reset()
            done = False
            episode_loss = 0

            while not done:
                pause_loss = self.env.latency + self.env.latency_accumulator[s]
                losses = [pause_loss, self.env.get_loss()]

                # if np.random.rand() < 0.1:
                #     a = np.random.choice(self.env.na)
                # else:
                # a = np.argmin(cumulative_loss[s] + losses)
                a = np.argmin(cumulative_loss[s])
                # if t == 1:
                #     print("s", s, "cum loss:", cumulative_loss[s])
                #     print("losses:", losses)
                    
                s_prime, loss, done = self.env.step(a)

                cumulative_loss[s, a] += loss
                episode_loss += loss

                s = s_prime
            
            self.loss_history[t] = episode_loss

        return self.loss_history
    

    def plot_loss(self, losses, title, xaxis_title, yaxis_title, window=500, min_episode=1):

        episodes = np.arange(min_episode, min_episode + len(losses))
        window_size = window

        rolling = pd.Series(losses).rolling(window=window_size, min_periods=1)
        smoothed_losses = rolling.mean()
        std_dev_losses = rolling.std()

        sns.set_theme(style='white')

        plt.figure(figsize=(10,6))
        plt.plot(episodes,
                 losses,
                 label='',
                 alpha=0.5,
                 color='royalblue')
        plt.plot(episodes,
                 smoothed_losses,
                 color='red',
                 label=f'Average (window = {window_size})')
        plt.fill_between(episodes,
                         smoothed_losses - std_dev_losses,
                         smoothed_losses + std_dev_losses,
                         color='blue',
                         alpha=0.2,
                         label='±1 Std Dev')
        
        plt.title(title, fontsize=14)
        plt.xlabel(xaxis_title, fontsize=12)
        plt.ylabel(yaxis_title, fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()