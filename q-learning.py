from single_process_mdp import SingleProcessMDP
import numpy as np

class QLearn:
    """
    QLearn
    """
    def __init__(self, env, epsilon = 0.1, gamma = 0.95, episodes = 100):
        self.env = env
        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma      # discount factor
        self.episodes = episodes

    def reset(self):
        return self.env.reset()

    def choose_action(self, q, s):
        if np.random.rand () < self.epsilon:
            return np.random.choice(self.env.na)
        return np.argmax(q[s])

    def execute(self):
        """
        Executes Q-learning.
        """
        s = self.reset()
        counts = np.zeros((self.env.ns, self.env.na))
        q = np.random.rand(self.env.ns, self.env.na) # Initialize Q-value table
        q_history = np.zeros((self.episodes, self.env.ns, self.env.na))

        for t in range(self.episodes):

            a = self.choose_action(q, s)    # Choose action
            counts[s, a] += 1

            s_prime, r = self.env.step(a)   # Perform action -> s_t+1 and r_t

            delta = r + self.gamma * (np.max(q[s_prime], axis=0)) - q[s, a]
            alpha = 2 / (counts[s, a]**(2/3) + 1)

            q[s, a] += alpha * delta
            q_history[t] = q

            s = s_prime # Update state
    
        return q_history


EPISODES = 100000
NUM_STATES = 10
env = SingleProcessMDP(NUM_STATES)
env.reset()
q_learn = QLearn(env, episodes=EPISODES)
hist = q_learn.execute()

optimal_q = hist[-1] # Optimal is the last one we stopped on.
print("Optimal Q:", optimal_q)

optimal_policy = np.argmax(optimal_q, axis=1)
print("Optimal Policy:", optimal_policy)