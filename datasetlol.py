
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
file = "datasetLoL\\matchinfo.csv"
dataset = pd.read_csv(file)
print(dataset.head())
dataset.groupby('Year').gamelength.mean().plot(kind='bar')
plt.show()