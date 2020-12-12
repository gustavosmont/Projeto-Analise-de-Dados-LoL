import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
file = "datasetLoL\\matchinfo.csv"
dataset = pd.read_csv(file)
plt.show(dataset.groupby('Year').mean().plot(kind='bar'))
