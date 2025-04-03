# visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load anomaly results
df = pd.read_csv("anomaly_results.csv")

# Scatter plot for anomalies
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df.index, y=df.iloc[:, 0], hue=df["anomaly"], palette={1: 'blue', -1: 'red'})
plt.title("Anomaly Detection (Red = Anomaly, Blue = Normal)")
plt.xlabel("Data Points")
plt.ylabel("Feature Value")
plt.show()
