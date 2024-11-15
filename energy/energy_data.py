import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EnergyData():
    def __init__(self, filename, reward_min, reward_max):
        file = pd.read_csv(filename, sep=',')
        self.data = file['Carbon Intensity gCO₂eq/kWh (direct)'].to_numpy()
        self.data_normalized = np.interp(self.data, 
                                         (np.min(self.data), np.max(self.data)), 
                                         (reward_min, reward_max))

    def sample_data(self, size):
        return np.random.choice(self.data, size=size, replace=True)

    def sample_normalized_data(self, size):
        return np.random.choice(self.data_normalized, size=size, replace=True)
    
    def plot_data(self, data=None, title=None):
        if data is None or data.size == 0:
            data = self.data

        plt.hist(data, bins=40, density=True, color='g')
        plt.ylabel("Density") # Use 'Density' if Density=True
        plt.xlabel("gCO₂eq/kWh")
        if not title:
            plt.title("Distribution of Carbon Intensities in 2023")
        plt.show()
        



    