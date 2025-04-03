# evaluation.py
import pandas as pd
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Dataset, Reader, SVD

# Load dataset
ratings = pd.read_csv("ratings.csv")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Load trained model
import pickle
model = pickle.load(open("recommendation_model.pkl", "rb"))

# Get predictions
predictions = model.test(testset)

# Compute RMSE
rmse_score = accuracy.rmse(predictions)

print(f"Evaluation Complete! RMSE: {rmse_score:.4f}")
