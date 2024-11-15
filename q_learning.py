import numpy as np
import matplotlib.pyplot as plt

class QLearn:
    """
    QLearn
    """
    def __init__(self, env, epsilon = 0.1, gamma = 0.95, episodes = 100):
        self.env = env
        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma      # discount factor
        self.episodes = episodes
        self.q_history = None
        self.optimal_policy = None
        self.optimal_q = None

        self.total_rewards = np.zeros((self.episodes))
        self.mse_loss = np.empty((self.episodes))
        self.mae_loss = np.empty((self.episodes))

    def reset(self):
        self.q_history = None
        self.optimal_policy = None
        s = self.env.reset()
        return s

    def choose_action(self, q, s):
        if np.random.rand () < self.epsilon:
            return np.random.choice(self.env.na)
        else:
            return np.argmax(q[s])
        

    def execute(self):
        counts = np.zeros((self.env.ns, self.env.na))
        q = np.random.rand(self.env.ns, self.env.na) # Initialize Q-value table
        q_history = np.zeros((self.episodes, self.env.ns, self.env.na))

        for t in range(self.episodes):
            s = self.env.reset() # Reset the MDP in each episode

            done = False

            curr_reward = 0
            while not done: # Repeat until the job completes
                a = self.choose_action(q, s)

                counts[s, a] += 1

                # Perform action -> get new state, reward (loss), and done flag
                s_prime, reward, is_done = self.env.step(a)
                curr_reward += reward

                # Update Q-Value
                delta = reward + self.gamma * (np.min(q[s_prime], axis=0)) - q[s, a]
                alpha = 2 / (counts[s, a]**(2/3) + 1)
                q[s, a] = q[s, a] + (alpha * delta)
                q_history[t] = q

                # Move to next state
                s = s_prime
                done = is_done

                # Track rewards
                self.total_rewards[t] = curr_reward

        self.q_history = q_history
        self.optimal_q = q_history[-1]
        self.optimal_policy = np.argmin(self.optimal_q, axis=1) # Minimize losses!
        # return q_history

    
    def compute_loss(self):
        for i in range(self.episodes):
            diff = self.q_history[-1] - self.q_history[i]

            infnorm = np.linalg.norm(diff, ord=np.inf)
            self.mae_loss[i] = infnorm
            self.mse_loss[i] = np.mean(diff ** 2)

    def show_optimal(self):
        print("Optimal Q:", self.optimal_q)
        print("Optimal Policy:", self.optimal_policy)

    def show_plots(self):
        self.plot_error(True, False)
        self.plot_rewards()


    def plot_error(self, plot_mae=False, plot_mse=True):
        self.compute_loss()
        if plot_mae:
            plt.plot(self.mae_loss, label="Mean Absolute Error")
        if plot_mse:
            plt.plot(self.mse_loss, label="Mean Squared Error")

        plt.xlabel("Episode")
        plt.ylabel("Error")
        plt.title(f"Q-Learning Error against {self.env.name}")

        plt.legend()

        plt.show()

    def plot_rewards(self):
        plt.plot(self.total_rewards)
        plt.title("Average reward")
        plt.show()
