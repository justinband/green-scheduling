from single_process_mdp import SingleProcessMDP
import numpy as np

class PI():
    """
    
    """

    def __init__(self, env, gamma=0.9):
        self.gamma = gamma
        self.env = env


    def reset(self):
        self.env.reset()

    def execute(self):
        """
        Initialize some random. We then evaluate the policy, then improve
        the policy. Evaluation of the policy depends on the value of the policy.
        """
        policy0 = np.random.randint(self.env.na, size=self.env.ns)
        policy1 = np.zeros(self.env.ns, dtype=int)
        n = 0

        policy_unchanged = False
        while not policy_unchanged:
            
            p_pi = np.array([[self.env.p[s, policy0[s], ss] for ss in range(self.env.ns)] for s in range(self.env.ns)])
            r_pi = np.array([self.env.r[s, policy0[s]] for s in range(self.env.ns)])

            # Policy evaluation, i.e. caluclate V!
            i = np.eye(self.env.ns)
            v0 = np.linalg.inv(i - self.gamma * p_pi) @ r_pi
            v1 = np.zeros(self.env.ns)

            # Policy improvement
            for s in range(self.env.ns):
                for a in range(self.env.na):
                   q = self.env.r[s, a] + self.gamma * sum([u * p for (u, p) in zip(v0, self.env.p[s, a])])
                   if (a == 0) or (q > v1[s]):
                       v1[s] = q
                       policy1[s] = a

            n += 1
            print(f"{n}: policy={policy1}, value={v1}")

            # Test if policy changed
            if (np.array(policy0) == np.array(policy1)).all():
                policy_unchanged = True
            else:
                policy0 = policy1
                policy1 = np.zeros(self.env.ns, dtype=int)

        return policy1


NUM_STATES = 10
env = SingleProcessMDP(NUM_STATES)
pi = PI(env)
pi.execute()
