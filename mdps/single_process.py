from energy.energy_data import EnergyData
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
    """
    def __init__(self, ns = 10):
        self.s = 0
        self.ns = ns
        self.na = 2
        self.pause = 0
        self.run = 1
        self.done = False
        self.name = "Single Process MDP"

        # Transition matrix
        self.p = np.zeros((self.ns, self.na, self.ns))
        for s in range(self.ns):
            self.p[s, self.pause, s] = 1 # Pause keeps same state
            if s < self.ns - 1: # Run moves to next state
                self.p[s, self.run, s+1] = 1
            else:
                self.p[s, self.run, s] = 1 # Final state, running keeps state

        # Reward matrix
        self.r = np.ones((self.ns, self.na))
        self.r[self.ns-1, self.run] = 0
        # FIXME: While this is bounded in [0, 1] we want to compare it to the energy
        #   costs, which are bounded [0, 1]. However, if we use loss=1 at each point it
        #   grows a lot.
        self.r[:self.ns-1, self.run] = 1/(self.ns) # Within [0,1]

    def step(self, action):
        """
        Performs an action in the MDP

        Early stopping. If in last state we simply stay there, getting no reward.
        """
        loss = self.r[self.s, action]

        if ((action == self.run) and (self.s == self.ns-1)) or self.done:
            self.done = True
            return self.s, loss, self.done
        else:
            new_state = np.random.choice(np.arange(self.ns), p=self.p[self.s, action])
            self.s = new_state
            self.done = False
            return new_state, loss, self.done

    def reset(self):
        """
        Resets the environment
        """
        self.s = 0
        self.done = False
        return self.s


class SingleProcessCostsMDP():
    """
    Create an MDP that simulates a single process. The length of the process is set by the
    number of states. There are only two actions, "run" and "pause", 1 and 0 respectively.

    Attributes
    ----------
    nS : int
        Number of states
    nA : int
        Number of actions. Set to 2.
    P : nS x nA x nS array
        Transition array
    """
    def __init__(self, ns = 10):
        self.s = 0
        self.ns = ns
        self.na = 2
        self.pause = 0
        self.run = 1
        self.done = False
        self.name = "Single Process with Costs MDP"
        self.loss_min = 0
        self.loss_max = 1

        self.energy = EnergyData('energy/DK_2023_hourly.csv',
                                 self.loss_min,
                                 self.loss_max)

        # Transition matrix
        self.p = np.zeros((self.ns, self.na, self.ns))
        for s in range(self.ns):
            self.p[s, self.pause, s] = 1 # Pause keeps same state
            if s < self.ns - 1: # Run moves to next state
                self.p[s, self.run, s+1] = 1
            else:
                self.p[s, self.run, s] = 1 # Final state, running keeps state

        # Reward Matirx
        self.r = np.ones((self.ns, self.na))
        self.r[self.ns-1, self.run] = 0 # TODO: Incorporate running cost to complete too

    def step(self, action):
        """
        Performs an action in the MDP

        Early stopping applied. i.e. In last state and run action is called then
        we stop and the max reward is given.
        """
        if ((action == self.run) and (self.s == self.ns-1)) or self.done:
            loss = self.r[self.s, action]
            self.done = True
            return self.s, loss, self.done
        else:
            loss = self.energy.sample_normalized_data(1)[0]
            # loss = self.r[self.s, action]
            self.s = np.random.choice(np.arange(self.ns), p=self.p[self.s, action])
            self.done = False
            return self.s, loss, self.done

    def reset(self):
        """
        Resets the environment
        """
        self.s = 0
        self.done = False
        return self.s

