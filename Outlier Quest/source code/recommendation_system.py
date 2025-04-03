# recommendation_system.py
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Load ratings dataset (user-item interactions)
ratings = pd.read_csv("ratings.csv")

# Convert to Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split dataset
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)

# Evaluate model
predictions = model.test(testset)
rmse_score = rmse(predictions)

# Save trained model
import pickle
pickle.dump(model, open("recommendation_model.pkl", "wb"))

print(f"Recommendation system trained! RMSE: {rmse_score:.4f}")
