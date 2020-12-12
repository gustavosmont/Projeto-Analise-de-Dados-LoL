import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
file = "datasetLoL\\kills.csv"
dataset = pd.read_csv(file)
print(dataset.columns)
print(dataset.head())
plt.scatter(dataset.x_pos,dataset.y_pos)


