import pandas as pd
import random
from datetime import datetime, timedelta

# 1. Load your existing movies and music
# Added on_bad_lines='skip' to bypass the malformed row on line 28 of movie.csv
movies_df = pd.read_csv('movie.csv', on_bad_lines='skip')
music_df = pd.read_csv('music.csv', on_bad_lines='skip')

print("✅ Data loaded successfully!")
print(f"📊 Movies columns: {movies_df.columns.tolist()}")
print(f"📊 Music columns: {music_df.columns.tolist()}")

# 2. Define users and their mood preferences
users = ['user1', 'user2', 'user3', 'user4', 'user5']
moods = ['sad', 'happy', 'motivated', 'romantic']

# 3. Create synthetic ratings
data = []
TOTAL_RATINGS = 800  # Generate 800 fake ratings

print(f"🎲 Generating {TOTAL_RATINGS} synthetic ratings...")

for _ in range(TOTAL_RATINGS):
    user = random.choice(users)
    user_pref_mood = random.choice(moods) 
    
    # 50% chance to pick a movie, 50% for music
    if random.random() > 0.5:
        item = movies_df.sample(1).iloc[0]
        item_type = 'movie'
    else:
        item = music_df.sample(1).iloc[0]
        item_type = 'music'
        
    # Try to find the title column safely (handles 'title', 'movie_title', 'name', etc.)
    title_col = 'title' if 'title' in item else ('movie_title' if 'movie_title' in item else item.index[0])
    item_title = str(item[title_col])
    
    # Try to find the mood column safely
    mood_col = 'mood' if 'mood' in item else item.index[-1]
    item_mood = str(item[mood_col]).lower()
    
    # 4. Assign a rating based on logic (This creates the pattern SVD will learn)
    # If the item matches the user's preferred mood, give a high rating (4 or 5)
    # Otherwise, give a lower rating (1, 2, or 3)
    if user_pref_mood in item_mood:
        rating = random.choices([4, 5], weights=[0.3, 0.7])[0] # Mostly 5s, some 4s
    else:
        rating = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0] # Mostly 2s and 3s
        
    data.append({
        'user_id': user,
        'item_title': item_title,
        'item_type': item_type,
        'rating': rating,
        'mood': user_pref_mood,
        'timestamp': datetime.now() - timedelta(days=random.randint(1, 30))
    })

# 5. Save to Excel
new_feedback = pd.DataFrame(data)
new_feedback.to_excel('feedback.xlsx', index=False)
print(f"✅ Successfully generated {TOTAL_RATINGS} synthetic ratings and updated feedback.xlsx!")
print("🚀 You can now run 'python evaluate.py' to check your new accuracy.")