import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class QLearn:
    """
    QLearn
    """
    def __init__(self, env, epsilon = 0.1, gamma = 1, episodes = 100, alpha = 1e4):
        self.env = env
        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma      # discount factor
        self.episodes = episodes
        self.q_history = None
        self.optimal_policy = None
        self.optimal_q = None

        self.total_rewards = np.zeros((self.episodes))
        self.energy_consumed = np.empty((self.episodes))
        self.mse_loss = np.empty((self.episodes))
        self.mae_loss = np.empty((self.episodes))

    def reset(self):
        self.q_history = None
        self.optimal_policy = None
        s = self.env.reset()
        return s

    def choose_action(self, q, s, t):
        if np.random.rand() < self.epsilon:
            a = np.random.choice(self.env.na)
            if t == 100:
                print("RANDOM! ", a)
            return a
        else:
            # Choose min?
            return np.argmin(q[s])
        
    def q_update(self, s, a, s_prime, q, loss, counts):
        # Q(s, a) = Q(s, a) + alpha(loss + gamma(Q(s, a) - min_a(Q(s', a))))
        # FIXME: Should it be:
        delta = loss + self.gamma * np.min(q[s_prime]) - q[s, a]
        # delta = loss + self.gamma * (q[s, a] - np.min(q[s_prime], axis=0))
        alpha = 0.000001
        # alpha = 2 / (counts[s, a]**(2/3) + 1)
        return q[s, a] + (alpha * delta)
                

    def execute(self):
        counts = np.zeros((self.env.ns, self.env.na))
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
                    q_copy[s, a] = self.q_update(s, a, s_prime, q_copy, loss, counts)
                

                a = self.choose_action(q, s, t) # choose best action
                s, loss, done = self.env.step(a) # perform action

                q[s, a] = q_copy[s, a] # Update with chosen action Q val

                # Update trackers
                episode_loss += loss
                counts[s, a] += 1
                q_history[t] = q

                if t==100:
                    print(f's: {s}, a: {a}, Q_p: {q[s, self.env.pause]}, Q_r: {q[s, self.env.run]}')

            self.total_rewards[t] = episode_loss

        self.q_history = q_history
        self.optimal_q = q_history[-1]
        self.optimal_policy = np.argmin(self.optimal_q, axis=1) # Minimize losses!


    # def execute(self):
    #     counts = np.zeros((self.env.ns, self.env.na))
    #     q = np.random.rand(self.env.ns, self.env.na) # Initialize Q-value table
    #     q_history = np.zeros((self.episodes, self.env.ns, self.env.na))

    #     for t in tqdm(range(self.episodes)):
    #         s = self.env.reset() # Reset the MDP in each episode

    #         done = False

    #         curr_reward = 0
    #         while not done: # Repeat until the job completes
    #             a = self.choose_action(q, s)

    #             counts[s, a] += 1

    #             # Perform action -> get new state, reward (loss), and done flag
    #             s_prime, reward, is_done = self.env.step(a)
    #             curr_reward += reward

    #             # Update Q-Value
    #             delta = reward + self.gamma * (np.min(q[s_prime], axis=0)) - q[s, a]
    #             alpha = 2 / (counts[s, a]**(2/3) + 1)
    #             q[s, a] = q[s, a] + (alpha * delta)
    #             q_history[t] = q

    #             # Move to next state
    #             s = s_prime
    #             done = is_done

    #             # Track rewards
    #             self.total_rewards[t] = curr_reward

    #     self.q_history = q_history
    #     self.optimal_q = q_history[-1]
    #     self.optimal_policy = np.argmin(self.optimal_q, axis=1) # Minimize losses!
    #     # return q_history

    
    def compute_loss(self):
        for i in range(self.episodes):
            # TODO: Difference in policy. Make it apparent.
            diff = self.q_history[-1] - self.q_history[i]

            infnorm = np.linalg.norm(diff, ord=np.inf)
            self.mae_loss[i] = infnorm
            self.mse_loss[i] = np.mean(diff ** 2)

    def show_optimal(self):
        print("Optimal Q:", self.optimal_q)
        print("Optimal Policy:", self.optimal_policy)

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

    def plot_rewards(self, title, xaxis_title, yaxis_title):
        plt.plot(self.total_rewards)
        plt.title(title)
        plt.xlabel(xaxis_title)
        plt.ylabel(yaxis_title)
        plt.show()
