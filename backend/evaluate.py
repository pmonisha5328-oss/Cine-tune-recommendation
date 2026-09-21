import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.metrics import mean_absolute_error

# 1. Load the data
df = pd.read_excel('feedback.xlsx')

# 2. Create a user-item matrix
matrix = df.pivot_table(index='user_id', columns='item_title', values='rating')

# 3. Calculate the global mean (e.g., 3.5 stars)
global_mean = df['rating'].mean()
print(f"📊 Global Mean Rating: {global_mean:.2f}")

# 4. Mean Centering: Subtract global mean, then fill missing with 0
matrix_centered = matrix - global_mean
matrix_filled = matrix_centered.fillna(0)

# 5. Perform SVD (k=3 for a slightly better fit)
U, sigma, Vt = svd(matrix_filled)
k = 3 
sigma_k = np.diag(sigma[:k])
U_k = U[:, :k]
Vt_k = Vt[:k, :]

# 6. Reconstruct the matrix and add the global mean back
predicted_matrix = np.dot(np.dot(U_k, sigma_k), Vt_k)
predicted_df = pd.DataFrame(predicted_matrix + global_mean, columns=matrix.columns, index=matrix.index)

# 7. Calculate Accuracy
actual_ratings = []
predicted_ratings = []

for user in matrix.index:
    for item in matrix.columns:
        if not pd.isna(matrix.loc[user, item]):
            actual_ratings.append(matrix.loc[user, item])
            predicted_ratings.append(predicted_df.loc[user, item])

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(actual_ratings, predicted_ratings)

# Calculate "Accuracy Percentage" (Within 1 Star)
correct_predictions = 0
for actual, predicted in zip(actual_ratings, predicted_ratings):
    if abs(actual - predicted) <= 1.0: 
        correct_predictions += 1

accuracy_percentage = (correct_predictions / len(actual_ratings)) * 100

print(f"📊 Model MAE (Mean Absolute Error): {mae:.4f}")
print(f"🎯 Model Accuracy (Within 1 Star): {accuracy_percentage:.2f}%")