"""
This file defines a single process MDP.
"""

import numpy as np


class SingleProcessMDP():
    """
    Create a MDP that simulates a single process. The length of the process is set by the
    number of states. There are only two actions, "run" and "pause", 1 and 0 respectively.

    Attributes
    ----------
    nS : int
        Number of states
    nA : int
        Number of actions. Set to 2.
    P : nS x nA x nS array
        Transition array

    Methods
    -------
    step(
    """
    def __init__(self, ns = 10):
        self.s = 0
        self.ns = ns
        self.na = 2
        self.PAUSE = 0
        self.RUN = 1

        def create_transitions():
            p = np.zeros((self.ns, self.na, self.ns))
            
            for s in range(self.ns):
                p[s, self.PAUSE, s] = 1 # Pause keeps same state
                if s < self.ns - 1: # Run moves to next state
                    p[s, self.RUN, s+1] = 1
                else:
                    p[s, self.RUN, s] = 1 # Final state, running keeps state

            return p

        def create_rewards():
            r = np.zeros((self.ns, self.na))
            r[self.ns - 1, self.RUN] = 1

            for s in range(self.ns-1):
                r[s, self.PAUSE] = -0.5
            return r
    
        self.p = create_transitions()
        self.r = create_rewards()

    
    def step(self, action):
        """
        Performs an action in the MDP
        """
        new_state = np.random.choice(np.arange(self.ns), p=self.p[self.s, action])
        reward = self.r[self.s, action]
        self.s = new_state
        return new_state, reward


    def reset(self):
        """
        Resets the environment
        """
        self.s = 0
        return self.s
