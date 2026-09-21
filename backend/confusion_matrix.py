import pandas as pd
import numpy as np
from numpy.linalg import svd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df = pd.read_excel('feedback.xlsx')
matrix = df.pivot_table(index='user_id', columns='item_title', values='rating')

# 2. Mean Centering & SVD
global_mean = df['rating'].mean()
matrix_centered = matrix - global_mean
matrix_filled = matrix_centered.fillna(0)

U, sigma, Vt = svd(matrix_filled)
k = 3
sigma_k = np.diag(sigma[:k])
U_k = U[:, :k]
Vt_k = Vt[:k, :]

predicted_matrix = np.dot(np.dot(U_k, sigma_k), Vt_k)
predicted_df = pd.DataFrame(predicted_matrix + global_mean, columns=matrix.columns, index=matrix.index)

# 3. Extract Actual vs Predicted for Classification
actual_ratings = []
predicted_ratings = []

for user in matrix.index:
    for item in matrix.columns:
        if not pd.isna(matrix.loc[user, item]):
            actual_ratings.append(int(matrix.loc[user, item]))
            # Round predicted ratings to nearest whole number (1-5)
            pred_rounded = int(round(predicted_df.loc[user, item]))
            # Clip values to be between 1 and 5 (in case rounding goes to 0 or 6)
            pred_rounded = max(1, min(5, pred_rounded))
            predicted_ratings.append(pred_rounded)

# 4. Generate Confusion Matrix
labels = [1, 2, 3, 4, 5]
cm = confusion_matrix(actual_ratings, predicted_ratings, labels=labels)

# 5. Calculate Exact Match Accuracy
exact_accuracy = accuracy_score(actual_ratings, predicted_ratings)
print(f"🎯 Exact Match Accuracy: {exact_accuracy * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(actual_ratings, predicted_ratings, labels=labels, zero_division=0))

# 6. Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Number of Ratings'})

plt.title('CineTune Rating Prediction Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Rating (Stars)', fontsize=12)
plt.ylabel('Actual Rating (Stars)', fontsize=12)
plt.tight_layout()

# Save the image for your report
plt.savefig('confusion_matrix.png', dpi=300)
print("\n✅ Confusion matrix saved as 'confusion_matrix.png'")
plt.show()