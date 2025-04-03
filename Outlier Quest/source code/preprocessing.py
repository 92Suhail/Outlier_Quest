# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    # Load dataset (replace 'data.csv' with actual dataset path)
    df = pd.read_csv('data.csv')
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Standardize numeric features
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    
    print("Data successfully loaded and preprocessed!")
    return df

if __name__ == "__main__":
    dataset = load_data()
    dataset.to_csv("processed_data.csv", index=False)
