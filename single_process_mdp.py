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
        self.pause = 0
        self.run = 1
        self.done = False

        self.p = np.zeros((self.ns, self.na, self.ns))
        for s in range(self.ns):
            self.p[s, self.pause, s] = 1 # Pause keeps same state
            if s < self.ns - 1: # Run moves to next state
                self.p[s, self.run, s+1] = 1
            else:
                self.p[s, self.run, s] = 1 # Final state, running keeps state


        self.r = np.zeros((self.ns, self.na))
        # energy = -0.5
        self.r[self.ns-1, self.run] = 10
        self.r[:self.ns-1, self.run] = 0.1
        # self.r[:self.ns-1, self.pause] = energy        # Pausing cost (Idle cost)
        # r[self.ns-1, self.run] = 1 * self.ns  # Completion reward

    def step(self, action):
        """
        Performs an action in the MDP

        Early stopping. If in last state we simply stay there, getting no reward.
        """
        reward = self.r[self.s, action]

        if ((action == self.run) and (self.s == self.ns-1)) or self.done:
            self.done = True
            return self.s, reward, self.done
        else:
            new_state = np.random.choice(np.arange(self.ns), p=self.p[self.s, action])
            self.s = new_state
            self.done = False
            return new_state, reward, self.done

    def reset(self):
        """
        Resets the environment
        """
        self.s = 0
        self.done = False
        return self.s
