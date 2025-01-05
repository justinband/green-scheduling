import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EnergyData():
    def __init__(self, filename, min=0, max=1):
        self.raw_data = pd.read_csv(filename, sep=',')
        self.data = self.raw_data['Carbon Intensity gCO₂eq/kWh (direct)'].to_numpy()
        self.data_normalized = np.sort(np.interp(self.data, 
                                         (np.min(self.data), np.max(self.data)), 
                                         (min, max)))
        self.mean = np.mean(self.data)
        self.sd = np.std(self.data)
        self.lower_bound, self.upper_bound = self.calc_iqr()

    def calc_iqr(self):
        q1 = np.percentile(self.data, 25)
        q3 = np.percentile(self.data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return lower_bound, upper_bound
        

    def sample_data(self, size):
        return np.random.choice(self.data, size=size, replace=True)

    def sample_normalized_data(self, size):
        return np.random.choice(self.data_normalized, size=size, replace=True)

    def get_data(self):
        return self.data_normalized
    
    def find_outliers(self):        
        return self.raw_data[self.raw_data['Carbon Intensity gCO₂eq/kWh (direct)'] >= self.upper_bound]

    def plot_outliers(self):
        outliers = self.find_outliers()
        datetimes = pd.to_datetime(outliers['Datetime (UTC)'].to_numpy())
        plt.hist(datetimes, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel("Date")
        plt.ylabel("Frequency")
        plt.title("When carbon intensities are above our upper bound")
        plt.tight_layout()
        plt.show()
    
    def plot_data(self, data=None, title=None, density=True):
        if data is None or data.size == 0:
            data = self.data

        sns.set_theme(style='whitegrid') # Set seaborn style


        plt.hist(data,
                 bins=100,
                 density=density,
                 color='royalblue',
                 edgecolor='black',
                 alpha=0.8)

        
        # Labels and title
        plt.ylabel("Density" if density else "Frequency", fontsize=12) # Use 'Density' if Density=True
        plt.xlabel(r"gCO$_2$eq/kWh", fontsize=12)
        plt.title(title if title else "Distribution of Carbon Intensities in 2023", fontsize=14)
        
        # Mean and Std Dev lines
        plt.axvline(self.mean, color='darkred', linestyle='solid', linewidth=1.2, label="Mean")
        # plt.axvline(self.mean + self.sd, color='green', linestyle='dashed', linewidth=1.2, label="+1 SD")
        # plt.axvline(self.mean - self.sd, color='purple', linestyle='dashed', linewidth=1.2, label='-1 SD')

        # IQR Bounds
        plt.axvline(self.lower_bound, color='purple', linestyle='dashed', linewidth=1.2, label='Lower Bound')
        plt.axvline(self.upper_bound, color='green', linestyle='dashed', linewidth=1.2, label='Upper Bound')

        # plt.text(self.mean, plt.ylim()[1]*0.9, 'Mean', color='darkred', fontsize=10, ha='center')
        # plt.text(self.mean + self.sd, plt.ylim()[1]*0.8, '+1 SD', color='darkgreen', fontsize=10, ha='center')
        # plt.text(self.mean - self.sd, plt.ylim()[1]*0.8, '-1 SD', color='purple', fontsize=10, ha='center')

        # Convert y-axis to percentage sign
        # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
        plt.legend()
        plt.show()

        print(f"Mean: {self.mean}")
        print(f"SD: {self.sd}")
        print(f"+1 SD: {self.mean + self.sd}")
        print(f"-1 SD: {self.mean - self.sd}")
        



    