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
        self.pause_action = 0
        self.run_action = 1
        self.job_complete = False

        # Build transition matrix
        self.p = np.zeros((self.ns, self.na, self.ns)) # nS x 2 x nS
        for s in range(self.ns):
            self.p[s, self.pause_action, s] = 1

            if s < self.ns - 1:
                self.p[s, self.run_action, s+1] = 1
            elif s == self.ns - 1:
                self.p[s, self.run_action, s] = 1

            #TODO: What is ending criteria?
            # if s == self.ns - 1: # last state. 
            #     self.P[s, self.run_action, s+1]

        # Define rewards
        self.r = np.zeros((self.ns, self.na))
        self.r[self.ns - 1, self.run_action] = 1


    def step(self, action):
        """
        Performs an action in the MDP
        """
        new_state = np.random.choice(np.arange(self.ns), p=self.p[self.s, action])
        reward = self.r[self.s, action]

        # When we are at the final piece of a job and then decide to run the job
        #   we are then finished.
        # This is considered early stopping.
        if self.s == self.ns-1 and action == self.run_action:
            # We have completed! And we must also stop...
            self.job_complete = True
            new_state = new_state + 1

        self.s = new_state

        return new_state, reward, self.job_complete


    def reset(self):
        """
        Resets the environment
        """
        self.s = 0
        self.job_complete = False
        return self.s, self.job_complete
