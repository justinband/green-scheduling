import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

class QLearn:
    """
    QLearn
    """
    def __init__(self, env, epsilon = 0.1, gamma = 1, episodes = 100, alpha = None, beta = 0.01):
        self.env = env
        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma      # discount factor
        self.beta = beta
        if alpha is not None:
            self.alpha = 1e-6
        else:
            self.alpha = alpha

        self.episodes = episodes
        self.q_history = None
        self.optimal_policy = None

        self.total_rewards = np.zeros((self.episodes))
        self.energy_consumed = np.empty((self.episodes))
        self.mse_loss = np.empty((self.episodes))
        self.mae_loss = np.empty((self.episodes))


    def reset(self):
        self.q_history = None
        self.optimal_policy = None
        s = self.env.reset()
        return s

    def choose_action(self, q, s):
        if np.random.rand() < self.epsilon:
            a = np.random.choice(self.env.na)
            return a
        else:
            # Choose min?
            return np.argmin(q[s])
        
    def q_update(self, s, a, s_prime, q, loss, t, counts):
        # Q(s, a) = Q(s, a) + alpha(loss + gamma(Q(s, a) - min_a(Q(s', a))))
        delta = loss + self.gamma * np.min(q[s_prime]) - q[s, a]
        # delta = loss + self.gamma * (q[s, a] - np.min(q[s_prime], axis=0))
        # alpha = 2 / (counts[s, a]**(2/3) + 1)

        alpha = self.alpha
        # alpha = self.alpha / (1 + self.beta * t) # decaying alpha
        return q[s, a] + (alpha * delta)

    def execute_informed(self, seed=None):
        all_losses = np.zeros((self.episodes))

        if seed is not None:
            np.random.seed(seed)

        counts = np.zeros((self.env.ns, self.env.na))
        # q = np.zeros((self.env.ns + 1, self.env.na))  # Add terminal state
        q = np.random.rand(self.env.ns + 1, self.env.na)  # Add terminal state

        # q = np.random.rand(self.env.ns, self.env.na) # Initialize Q-value table
        q_history = np.zeros((self.episodes, self.env.ns + 1, self.env.na))

        for t in tqdm(range(self.episodes)):
            s = self.env.reset()
            done = False
            episode_loss = 0

            while not done:
                s_prime = self.env.get_next_state()
                actions = [self.env.pause, self.env.run]
                pause_loss = self.env.latency + self.env.latency_accumulator[s]
                losses = [pause_loss, self.env.get_loss()]

                # Last state is "done" state. Q = 0 is best.
                if s_prime == self.env.ns:
                    q[s_prime, :] = 0
                    # s_prime = self.env.ns - 1

                q_copy = q.copy() # make temp updates
                for (a, loss) in zip(actions, losses):
                    q_copy[s, a] = self.q_update(s, a, s_prime, q_copy, loss, t, counts)
                

                a = self.choose_action(q, s) # choose best action
                s, loss, done = self.env.step(a) # perform action

                q[s, a] = q_copy[s, a] # Update with chosen action Q val

                # Update trackers
                episode_loss += loss
                counts[s, a] += 1
                q_history[t] = q

            all_losses[t] = episode_loss

        self.q_history = q_history
        optimal_q = q_history[-1]
        optimal_policy = np.argmin(optimal_q, axis=1) # Minimize losses!
        self.total_rewards = all_losses
        return all_losses, optimal_policy


    def execute_uninformed(self, seed=None):
        losses = np.zeros((self.episodes))

        if seed is not None:
            np.random.seed(seed)

        counts = np.zeros((self.env.ns, self.env.na))
        q = np.random.rand(self.env.ns, self.env.na) # Initialize Q-value table
        q_history = np.zeros((self.episodes, self.env.ns, self.env.na))

        for t in tqdm(range(self.episodes)):
            s = self.env.reset() # Reset the MDP in each episode
            done = False
            episode_loss = 0

            while not done: # Repeat until the job completes
                a = self.choose_action(q, s)
                s_prime, loss, is_done = self.env.step(a) # perform action

                counts[s, a] += 1
                q[s, a] = self.q_update(s, a, s_prime, q, loss, t, counts)
            
                episode_loss += loss
                q_history[t] = q

                s = s_prime # Move to next state
                done = is_done

            losses[t] = episode_loss
            
        optimal_q = q_history[-1]
        optimal_policy = np.argmin(optimal_q, axis=1) # Minimize losses!
        return losses, optimal_policy

    def plot_losses(self,
                    losses,
                    title,
                    xaxis_title,
                    yaxis_title,
                    window=500,
                    min_episode=1,
                    show_std_dev=True
                    ):

        episodes = np.arange(min_episode, min_episode + len(losses))
        # episodes = np.arange(1,len(losses)+1)

        window_size = window  # Calculate trend line

        rolling = pd.Series(losses).rolling(window=window_size, min_periods=1)
        smoothed_losses = rolling.mean()
        std_dev_losses = rolling.std()

        sns.set_theme(style='white')

        plt.figure(figsize=(10, 6))
        plt.plot(episodes,
                 losses,
                 label='',
                 alpha=0.5,
                 color='royalblue')
        plt.plot(episodes,
                 smoothed_losses,
                 color='red',
                 label=f'Average (window = {window_size})')
        
        if show_std_dev:
            plt.fill_between(episodes,
                            smoothed_losses - std_dev_losses,
                            smoothed_losses + std_dev_losses,
                            color='blue',
                            alpha=0.2,
                            label='±1 Std Dev')

        # Add labels and legend
        plt.title(title, fontsize=16)
        plt.xlabel(xaxis_title, fontsize=12)
        plt.ylabel(yaxis_title, fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
