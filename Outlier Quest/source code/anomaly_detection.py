# anomaly_detection.py
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load preprocessed data
df = pd.read_csv("processed_data.csv")

# Train Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(df.iloc[:, :-1])

# Save results
df.to_csv("anomaly_results.csv", index=False)
print("Anomaly detection completed! Results saved in 'anomaly_results.csv'.")
