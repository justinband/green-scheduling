from energy.energy_data import EnergyData
import numpy as np

class MDPLatency():
    """
    This MDP simulates a single job.
    The job length is set by the number of states.
    There are two actions, 'run' and 'pause', 1 and 0 respectively.

    This MDP uses 0-1 loss. There is no reward at the end. Instead, accumulating latency
    is used.
    """
    def __init__(self, ns = 10):
        # MDP Definitions
        self.s = 0
        self.ns = ns
        self.na = 2
        self.done = False
        self.name = "Cost MDP with Latency"

        # Actions
        self.pause = 0
        self.run = 1


        # Energy
        self.latency = 0.1
        self.loss_min = 0
        self.loss_max = 1
        self.energy = EnergyData('energy/DK_2023_hourly.csv',
                                 self.loss_min,
                                 self.loss_max)   
        self.curr_loss = self.sample_energy()

        # Transition matrix
        self.p = np.zeros((self.ns, self.na, self.ns))
        for s in range(self.ns):
            self.p[s, self.pause, s] = 1 # Pause keeps same state
            if s < self.ns - 1: # Run moves to next state
                self.p[s, self.run, s+1] = 1
            else:
                self.p[s, self.run, s] = 1 # Final state, running keeps state

        # Latency is added for each pause action 
        self.latency_accumulator = np.zeros((self.ns))

    def get_next_state(self):
        """
        Returns the next state. This is built on the assumption that state transitions are known.
        """
        return self.s + 1
    
    def get_loss(self):
        """
        Returns the loss that will be used at this time step
        """
        return self.curr_loss
    
    def sample_energy(self):
        """
        Samples normalizaed energy data from the loaded data
        """
        return self.energy.sample_normalized_data(1)[0]

    def add_latency(self, state):
        self.latency_accumulator[state] += self.latency
        return self.latency_accumulator[state]


    def step(self, action):
        """
        Performs an action in the MDP

        Early stopping applied. i.e. In last state and run action is called then
        we stop and the max reward is given.
        """

        loss = self.curr_loss # Get current loss
        self.curr_loss = self.sample_energy() # Update loss with a new sample

        # We reset the MDP if ran in the last state.
        if ((action == self.run) and (self.s == self.ns-1)) or self.done:
            self.done = True
            return self.s, loss, self.done

        # In other states, we determine the loss based on the action.
        else:
            if action == self.pause:
                loss = self.add_latency(self.s)

            self.s = np.random.choice(np.arange(self.ns), p=self.p[self.s, action])
            return self.s, loss, self.done

    def reset(self):
        """
        Resets the environment
        """
        self.s = 0
        self.done = False
        self.latency_accumulator = np.zeros((self.ns))
        return self.s
