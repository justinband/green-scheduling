from mdps.single_process import SingleProcessMDP
import numpy as np

class VI():
    """
    VI Desc.
    """
    def __init__(self, env, gamma = 0.9):
        self.gamma = gamma
        self.env = env
        self.policy = np.zeros(self.env.ns)

    def reset(self):
        """
        Resets the policy and the environment
        """
        self.policy = np.zeros(self.env.ns)
        self.env.reset()

    def execute(self):
        """
        Initialize some random values for states. This function directly computes the optimal
        value function by iterating through the environment. The optimal value is the
        maximum expected reward for each state under some optimal policy. This function also finds
        the optimal policy for the value function.
        """
        self.reset()
        n = 0
        v0 = np.array([1/(1-self.gamma) for _ in range(self.env.ns)])
        v1 = np.zeros(self.env.ns)

        epsilon = ((10**-2) * (1 - self.gamma)) / (2 * self.gamma)
        while True:
            for s in range(self.env.ns):
                for a in range(self.env.na):
                    action_val = self.env.r[s, a] + self.gamma * sum([V * p for (V, p) in zip(v0, self.env.p[s, a])])
        
                    if action_val > v1[s]:
                        v1[s] = action_val
                        self.policy[s] = a

            n += 1
            print(f"{n}: policy={self.policy}, value={v1}")

            if np.linalg.norm(v1 - v0) < epsilon:
                return self.policy, v1
            else:
                v0 = v1.copy()
                v1.fill(0)


NUM_STATES = 10
env = SingleProcessMDP(NUM_STATES)
vi = VI(env)
vi.execute()
