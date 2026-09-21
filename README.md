# 🎬🎵 CineTune: Mood-Based Movie & Music Recommendation System

CineTune is a full-stack web application that recommends movies and music based on a user's current mood (Sad, Happy, Motivated, Romantic). It uses Machine Learning (SVD Collaborative Filtering) to provide personalized recommendations.

## ✨ Features
- **Mood Selection:** Users select their current mood to get tailored recommendations.
- **Collaborative Filtering:** Uses Singular Value Decomposition (SVD) to predict user ratings for unseen items.
- **Cold-Start Handling:** Gracefully falls back to mood-based top-rated items for new users.
- **Synthetic Data Generation:** Includes a script to generate 800+ realistic user ratings to train the model effectively.
- **Model Evaluation:** Includes scripts to calculate RMSE, MAE, and generate a Confusion Matrix.

## 🛠️ Tech Stack
- **Backend:** Python, Flask, Flask-CORS
- **Machine Learning:** Pandas, NumPy, Scikit-Learn, SVD
- **Frontend:** HTML, CSS, JavaScript
- **Data Storage:** CSV (Movies/Music), Excel (User Feedback)

## 📊 Model Performance
- **Exact Match Accuracy:** 67.11%
- **Accuracy (Within 1 Star):** 86.84%
- **Mean Absolute Error (MAE):** 0.39
- *Note: Accuracy is calculated using Mean Centering to handle sparse data effectively.*

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pmonisha5328-oss/Cine-tune-recommendation.git
   cd Cine-tune-recommendation/backend
