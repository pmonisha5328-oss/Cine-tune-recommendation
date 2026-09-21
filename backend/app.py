from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import csv, os, random
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# --- Flask setup ---
app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📁 Data directory: {DATA_DIR}")

# Supported moods
SUPPORTED_MOODS = ['sad', 'happy', 'motivated', 'romantic']

def list_data_files():
    """List all data files in the directory for debugging"""
    print("📂 Files in data directory:")
    for file in os.listdir(DATA_DIR):
        if file.endswith(('.csv', '.xlsx', '.json')):
            file_path = os.path.join(DATA_DIR, file)
            size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"   - {file} ({size} bytes)")

def read_csv(filename):
    """Read CSV file and return list of dictionaries"""
    path = os.path.join(DATA_DIR, filename)
    print(f"📖 Attempting to read CSV: {path}")
    
    if not os.path.exists(path):
        print(f"❌ CSV file {filename} not found at {path}")
        return []
    
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            print(f"✅ Successfully read {len(data)} rows from {filename}")
            return data
    except Exception as e:
        print(f"❌ Error reading CSV {filename}: {e}")
        return []

def create_sample_feedback_excel():
    """Create a sample feedback Excel file if it doesn't exist or is corrupted"""
    excel_path = os.path.join(DATA_DIR, "feedback.xlsx")
    
    # Enhanced sample data with more users and ratings for both movies and music
    sample_data = {
        'user_id': ['user1', 'user2', 'user1', 'user3', 'user2', 'user3', 'user4', 'user5', 'user1', 'user2', 'user3', 'user4'],
        'item_title': ['Inception', 'Shape of You', 'The Dark Knight', 'Blinding Lights', 'Forrest Gump', 'Someone Like You', 
                      'Interstellar', 'Uptown Funk', 'Fight Club', 'Thinking Out Loud', 'The Godfather', 'Happy'],
        'item_type': ['movie', 'music', 'movie', 'music', 'movie', 'music', 'movie', 'music', 'movie', 'music', 'movie', 'music'],
        'rating': [5, 4, 5, 4, 5, 3, 4, 5, 4, 5, 5, 4],
        'mood': ['motivated', 'happy', 'motivated', 'motivated', 'happy', 'sad', 'motivated', 'happy', 'motivated', 'romantic', 'sad', 'happy'],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(12)]
    }
    
    try:
        df = pd.DataFrame(sample_data)
        df.to_excel(excel_path, index=False)
        print(f"✅ Created sample feedback Excel file with {len(df)} entries")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample Excel file: {e}")
        return False

def read_excel_feedback(filename):
    """Read feedback data from Excel file with enhanced error handling"""
    path = os.path.join(DATA_DIR, filename)
    print(f"📖 Attempting to read Excel: {path}")
    
    if not os.path.exists(path):
        print(f"❌ Excel file {filename} not found at {path}")
        if create_sample_feedback_excel():
            return read_excel_feedback(filename)
        return []
    
    try:
        file_size = os.path.getsize(path)
        print(f"📊 File size: {file_size} bytes")
        
        if file_size == 0:
            print("❌ Excel file is empty (0 bytes)")
            if create_sample_feedback_excel():
                return read_excel_feedback(filename)
            return []
            
        with zipfile.ZipFile(path, 'r') as zip_file:
            print("✅ File is a valid zip archive")
            
    except zipfile.BadZipFile:
        print("❌ File is not a valid Excel file (BadZipFile error)")
        if create_sample_feedback_excel():
            return read_excel_feedback(filename)
        return []
    
    try:
        df = pd.read_excel(path, engine='openpyxl')
        print(f"✅ Successfully loaded Excel with {len(df)} rows and {len(df.columns)} columns")
        print(f"📊 Columns: {list(df.columns)}")
        
        feedback_data = df.to_dict('records')
        print(f"✅ Converted to {len(feedback_data)} feedback entries")
        
        return feedback_data
    except Exception as e:
        print(f"❌ Error reading Excel file {filename}: {e}")
        if create_sample_feedback_excel():
            return read_excel_feedback(filename)
        return []

def write_user(data):
    """Write new user to CSV"""
    path = os.path.join(DATA_DIR, "users.csv")
    fieldnames = ["name", "email", "password"]
    exists = os.path.isfile(path)
    
    try:
        with open(path, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(data)
        print(f"✅ User {data['email']} saved successfully")
    except Exception as e:
        print(f"❌ Error saving user: {e}")

# --- LOAD DATA ---
print("🎬 Loading movies data...")
movies_data = read_csv("movie.csv")
if len(movies_data) == 0:
    print("🔍 Trying movies.csv instead...")
    movies_data = read_csv("movies.csv")
print(f"🎬 Total movies loaded: {len(movies_data)}")

print("🎵 Loading music data...")
music_data = read_csv("music.csv")
if len(music_data) == 0:
    print("🔍 Trying songs.csv instead...")
    music_data = read_csv("songs.csv")
print(f"🎵 Total songs loaded: {len(music_data)}")

# --- COLLABORATIVE FILTERING SYSTEM FOR BOTH MOVIES AND MUSIC ---
class CollaborativeFilteringSVD:
    def __init__(self):
        self.user_item_matrix = None
        self.item_titles = None
        self.user_ids = None
        self.predicted_ratings = None
        self.k = 2
        self.is_trained = False
        self.item_types = {}  # Track whether item is movie or music
        
    def build_matrix_from_excel(self, excel_filename):
        """Build user-item rating matrix from Excel feedback data"""
        print(f"🔧 Building collaborative filtering matrix from {excel_filename}...")
        feedback_data = read_excel_feedback(excel_filename)
        
        if not feedback_data:
            print(f"❌ No feedback data found in {excel_filename}")
            return False
            
        df = pd.DataFrame(feedback_data)
        print(f"📊 Raw feedback data: {len(df)} entries")
        
        required_columns = ['user_id', 'item_title', 'rating', 'item_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing columns in Excel file: {missing_columns}")
            return False
        
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating'])
        
        print(f"📊 Valid ratings after cleaning: {len(df)}")
        
        if len(df) < 5:
            print("⚠ Not enough ratings data for collaborative filtering (need at least 5)")
            return False
            
        try:
            # Create user-item matrix
            self.user_item_matrix = df.pivot_table(
                index='user_id', 
                columns='item_title', 
                values='rating', 
                fill_value=0
            )
            
            self.item_titles = self.user_item_matrix.columns.tolist()
            self.user_ids = self.user_item_matrix.index.tolist()
            
            # Store item types for filtering
            for _, row in df.iterrows():
                self.item_types[row['item_title']] = row['item_type']
            
            print(f"✅ Built collaborative matrix with {len(self.user_ids)} users and {len(self.item_titles)} items")
            print(f"👥 Users: {self.user_ids}")
            print(f"📊 Items by type: Movies - {len([v for v in self.item_types.values() if v == 'movie'])}, "
                  f"Music - {len([v for v in self.item_types.values() if v == 'music'])}")
            
            return True
        except Exception as e:
            print(f"❌ Error building matrix: {e}")
            return False
    
    def train_svd(self):
        """Train SVD model for collaborative filtering"""
        if self.user_item_matrix is None:
            print("❌ No user-item matrix to train on")
            return False
            
        print("🔧 Training SVD model for collaborative filtering...")
        
        R = self.user_item_matrix.values.astype(float)
        
        if R.shape[0] < 2 or R.shape[1] < 2:
            print("❌ Matrix too small for SVD")
            return False
        
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        
        print(f"📊 Matrix shape: {R_demeaned.shape}")
        
        k = min(self.k, min(R_demeaned.shape) - 1)
        if k <= 0:
            print("❌ Not enough data for SVD decomposition")
            return False
            
        print(f"🔧 Performing SVD with k={k}...")
        try:
            U, sigma, Vt = svds(R_demeaned, k=k)
            sigma = np.diag(sigma)
            
            all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
            self.predicted_ratings = pd.DataFrame(
                all_user_predicted_ratings, 
                index=self.user_ids, 
                columns=self.item_titles
            )
            
            self.is_trained = True
            print("✅ SVD collaborative filtering model trained successfully")
            return True
        except Exception as e:
            print(f"❌ SVD training failed: {e}")
            return False
    
    def get_recommendations(self, user_id, item_type=None, mood=None, n_recommendations=10):
        """Get collaborative filtering recommendations for a specific user"""
        if not self.is_trained or self.predicted_ratings is None:
            print("❌ No predictions available - collaborative filtering not trained")
            return []
            
        if user_id not in self.user_ids:
            print(f"❌ User {user_id} not found in training data")
            return self.get_popular_recommendations(item_type, mood, n_recommendations)
            
        print(f"🎯 Getting collaborative recommendations for user {user_id}, type: {item_type}, mood: {mood}")
        
        try:
            user_ratings = self.predicted_ratings.loc[user_id]
            
            # Get items user hasn't rated
            user_rated_items = self.user_item_matrix.loc[user_id]
            unrated_items = user_rated_items[user_rated_items == 0].index
            
            # Filter by item type and mood if specified
            filtered_items = []
            for item in unrated_items:
                # Filter by item type
                if item_type and self.item_types.get(item) != item_type:
                    continue
                
                # Filter by mood (this would require mood data for items)
                if mood:
                    # In a real system, you'd have mood data for each item
                    # For now, we'll use a simple approach
                    item_mood = self.get_item_mood(item, item_type)
                    if item_mood and item_mood != mood:
                        continue
                
                filtered_items.append(item)
            
            if not filtered_items:
                return self.get_popular_recommendations(item_type, mood, n_recommendations)
            
            # Get top recommendations
            recommendations = user_ratings[filtered_items].sort_values(ascending=False).head(n_recommendations)
            
            print(f"✅ Generated {len(recommendations)} collaborative recommendations")
            return recommendations.index.tolist()
            
        except Exception as e:
            print(f"❌ Error getting collaborative recommendations: {e}")
            return self.get_popular_recommendations(item_type, mood, n_recommendations)
    
    def get_item_mood(self, item_title, item_type):
        """Get mood for an item from the original data"""
        if item_type == 'movie':
            for movie in movies_data:
                if movie.get('title', '').lower() == item_title.lower():
                    return movie.get('mood', '').lower()
        elif item_type == 'music':
            for song in music_data:
                if song.get('title', '').lower() == item_title.lower():
                    return song.get('mood', '').lower()
        return None
    
    def get_popular_recommendations(self, item_type=None, mood=None, n_recommendations=10):
        """Get popular items as fallback"""
        if not self.user_item_matrix:
            return []
            
        # Calculate average ratings
        avg_ratings = self.user_item_matrix.mean()
        
        # Filter by item type and mood
        filtered_items = []
        for item in avg_ratings.index:
            if item_type and self.item_types.get(item) != item_type:
                continue
            
            if mood:
                item_mood = self.get_item_mood(item, self.item_types.get(item))
                if item_mood and item_mood != mood:
                    continue
            
            filtered_items.append(item)
        
        if not filtered_items:
            return []
            
        popular_items = avg_ratings[filtered_items].sort_values(ascending=False).head(n_recommendations)
        return popular_items.index.tolist()

# Initialize collaborative filtering system
cf_system = CollaborativeFilteringSVD()

def initialize_collaborative_filtering():
    """Initialize collaborative filtering system with SVD"""
    print("🚀 Initializing Collaborative Filtering with SVD...")
    
    excel_filenames = ['feedback.xlsx', 'ratings.xlsx', 'user_ratings.xlsx']
    
    for filename in excel_filenames:
        print(f"🔍 Trying {filename}...")
        if cf_system.build_matrix_from_excel(filename):
            if cf_system.train_svd():
                print(f"✅ Collaborative Filtering system initialized from {filename}")
                return True
            else:
                print(f"❌ SVD training failed with {filename}")
        else:
            print(f"❌ Could not build matrix from {filename}")
    
    print("⚠ Collaborative Filtering will be disabled - using content-based filtering only")
    return False

# List all data files at startup
list_data_files()

# Initialize collaborative filtering on startup
cf_initialized = initialize_collaborative_filtering()

# --- ROUTES ---
@app.route('/')
def index():
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory(app.template_folder, filename)

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    write_user(data)
    return jsonify({"status": "success", "message": "Signup successful!"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email, password = data["email"], data["password"]
    users_path = os.path.join(DATA_DIR, "users.csv")

    if not os.path.isfile(users_path):
        return jsonify({"status": "error", "message": "No users found."})

    with open(users_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for user in reader:
            if user["email"] == email and user["password"] == password:
                return jsonify({"status": "success", "message": "Login successful!"})
    return jsonify({"status": "error", "message": "Invalid credentials."})

@app.route('/recommend/movies', methods=['POST'])
def recommend_movies():
    data = request.json
    mood = data.get('mood', '').lower()
    user_id = data.get('user_id', 'default_user')
    use_collaborative = data.get('use_collaborative', True)
    
    print(f"🎬 Movie recommendations for mood: '{mood}', user: '{user_id}'")
    
    if not movies_data:
        return jsonify({"error": "Movie database not available"}), 500
    
    # Clean movie data
    cleaned_movies = []
    for movie in movies_data:
        cleaned_movie = {
            'title': movie.get('title', 'Unknown Title'),
            'mood': movie.get('mood', '').lower(),
            'language': movie.get('language', ''),
            'genre': movie.get('genre', ''),
            'rating': movie.get('rating', ''),
            'year': movie.get('year', ''),
            'movie_platform': movie.get('movie_platform', ''),
            'image_url': movie.get('image_url', ''),
            'description': movie.get('description', 'Description not available'),
            'director': movie.get('director', 'Director information not available'),
            'cast': movie.get('cast', 'Cast information not available'),
            'item_type': 'movie'
        }
        cleaned_movies.append(cleaned_movie)
    
    recommended_movies = []
    
    # Collaborative Filtering recommendations
    if use_collaborative and cf_system.is_trained:
        try:
            collaborative_recs = cf_system.get_recommendations(
                user_id, 'movie', mood, 8
            )
            print(f"🎯 Collaborative movie recommendations: {collaborative_recs}")
            
            movie_title_dict = {movie['title'].lower(): movie for movie in cleaned_movies}
            for movie_title in collaborative_recs:
                if movie_title.lower() in movie_title_dict:
                    movie_data = movie_title_dict[movie_title.lower()]
                    movie_data['recommendation_type'] = 'collaborative'
                    movie_data['recommendation_engine'] = 'svd_collaborative'
                    recommended_movies.append(movie_data)
                    
            print(f"✅ Found {len(recommended_movies)} collaborative movie recommendations")
        except Exception as e:
            print(f"❌ Collaborative movie recommendation error: {e}")
    
    # Content-based filtering for remaining slots
    remaining_slots = 10 - len(recommended_movies)
    if remaining_slots > 0:
        mood_movies = [movie for movie in cleaned_movies if not mood or movie['mood'] == mood]
        if mood_movies:
            content_based = random.sample(mood_movies, min(remaining_slots, len(mood_movies)))
            for movie in content_based:
                if movie not in recommended_movies:
                    movie['recommendation_type'] = 'content_based'
                    movie['recommendation_engine'] = 'mood_based'
                    recommended_movies.append(movie)
        else:
            fallback_movies = random.sample(cleaned_movies, min(remaining_slots, len(cleaned_movies)))
            for movie in fallback_movies:
                movie['recommendation_type'] = 'fallback'
                movie['recommendation_engine'] = 'random'
                recommended_movies.append(movie)
    
    # Ensure we have exactly 10 recommendations
    if len(recommended_movies) < 10:
        additional_needed = 10 - len(recommended_movies)
        available_movies = [m for m in cleaned_movies if m not in recommended_movies]
        if available_movies:
            additional_movies = random.sample(available_movies, min(additional_needed, len(available_movies)))
            for movie in additional_movies:
                movie['recommendation_type'] = 'additional'
                movie['recommendation_engine'] = 'fallback'
                recommended_movies.append(movie)
    
    print(f"🎉 Final movie recommendations: {len(recommended_movies)} movies")
    return jsonify(recommended_movies[:10])

@app.route('/recommend/music', methods=['POST'])
def recommend_music():
    data = request.json
    mood = data.get('mood', '').lower()
    user_id = data.get('user_id', 'default_user')
    use_collaborative = data.get('use_collaborative', True)
    
    print(f"🎵 Music recommendations for mood: '{mood}', user: '{user_id}'")
    
    if not music_data:
        return jsonify({"error": "Music database not available"}), 500
    
    # Clean music data
    cleaned_music = []
    for song in music_data:
        cleaned_song = {
            'title': song.get('title', 'Unknown Title'),
            'artist': song.get('artist', 'Unknown Artist'),
            'mood': song.get('mood', '').lower(),
            'language': song.get('language', ''),
            'genre': song.get('genre', ''),
            'album': song.get('album', ''),
            'duration': song.get('duration', ''),
            'music_platform': song.get('music_platform', ''),
            'preview_url': song.get('preview_url', ''),
            'item_type': 'music'
        }
        cleaned_music.append(cleaned_song)
    
    recommended_music = []
    
    # Collaborative Filtering recommendations
    if use_collaborative and cf_system.is_trained:
        try:
            collaborative_recs = cf_system.get_recommendations(
                user_id, 'music', mood, 8
            )
            print(f"🎯 Collaborative music recommendations: {collaborative_recs}")
            
            music_title_dict = {song['title'].lower(): song for song in cleaned_music}
            for song_title in collaborative_recs:
                if song_title.lower() in music_title_dict:
                    song_data = music_title_dict[song_title.lower()]
                    song_data['recommendation_type'] = 'collaborative'
                    song_data['recommendation_engine'] = 'svd_collaborative'
                    recommended_music.append(song_data)
                    
            print(f"✅ Found {len(recommended_music)} collaborative music recommendations")
        except Exception as e:
            print(f"❌ Collaborative music recommendation error: {e}")
    
    # Content-based filtering for remaining slots
    remaining_slots = 10 - len(recommended_music)
    if remaining_slots > 0:
        mood_music = [song for song in cleaned_music if not mood or song['mood'] == mood]
        if mood_music:
            content_based = random.sample(mood_music, min(remaining_slots, len(mood_music)))
            for song in content_based:
                if song not in recommended_music:
                    song['recommendation_type'] = 'content_based'
                    song['recommendation_engine'] = 'mood_based'
                    recommended_music.append(song)
        else:
            fallback_music = random.sample(cleaned_music, min(remaining_slots, len(cleaned_music)))
            for song in fallback_music:
                song['recommendation_type'] = 'fallback'
                song['recommendation_engine'] = 'random'
                recommended_music.append(song)
    
    # Ensure we have exactly 10 recommendations
    if len(recommended_music) < 10:
        additional_needed = 10 - len(recommended_music)
        available_music = [s for s in cleaned_music if s not in recommended_music]
        if available_music:
            additional_music = random.sample(available_music, min(additional_needed, len(available_music)))
            for song in additional_music:
                song['recommendation_type'] = 'additional'
                song['recommendation_engine'] = 'fallback'
                recommended_music.append(song)
    
    print(f"🎉 Final music recommendations: {len(recommended_music)} songs")
    return jsonify(recommended_music[:10])

@app.route('/rate/item', methods=['POST'])
def rate_item():
    """Endpoint to record user ratings for both movies and music"""
    data = request.json
    user_id = data.get('user_id', 'default_user')
    item_title = data.get('item_title')
    item_type = data.get('item_type')  # 'movie' or 'music'
    rating = data.get('rating')
    mood = data.get('mood', '')
    
    if not all([user_id, item_title, item_type, rating]):
        return jsonify({"status": "error", "message": "Missing required fields"}), 400
    
    if item_type not in ['movie', 'music']:
        return jsonify({"status": "error", "message": "Item type must be 'movie' or 'music'"}), 400
    
    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            return jsonify({"status": "error", "message": "Rating must be between 1 and 5"}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "Rating must be a number"}), 400
    
    # Create new rating entry
    new_rating = {
        'user_id': user_id,
        'item_title': item_title,
        'item_type': item_type,
        'rating': rating,
        'mood': mood,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Append to Excel file
    excel_path = os.path.join(DATA_DIR, "feedback.xlsx")
    
    try:
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path, engine='openpyxl')
            df = pd.concat([df, pd.DataFrame([new_rating])], ignore_index=True)
        else:
            df = pd.DataFrame([new_rating])
        
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"✅ Rating saved: {user_id} rated {item_type} '{item_title}' as {rating} (mood: {mood})")
        
        # Retrain collaborative filtering model with updated data
        print("🔄 Retraining collaborative filtering model with new data...")
        initialize_collaborative_filtering()
        
        return jsonify({
            "status": "success", 
            "message": f"Rating for {item_type} saved successfully",
            "next_recommendations": True
        })
        
    except Exception as e:
        print(f"❌ Error saving rating: {e}")
        return jsonify({"status": "error", "message": "Failed to save rating"}), 500

@app.route('/collaborative/status', methods=['GET'])
def collaborative_status():
    """Check collaborative filtering system status"""
    status = {
        'initialized': cf_system.is_trained,
        'users_count': len(cf_system.user_ids) if cf_system.user_ids else 0,
        'items_count': len(cf_system.item_titles) if cf_system.item_titles else 0,
        'matrix_shape': cf_system.user_item_matrix.shape if cf_system.user_item_matrix is not None else None,
        'recommendation_engine': 'SVD Collaborative Filtering'
    }
    return jsonify(status)

@app.route('/feedback/preview', methods=['GET'])
def feedback_preview():
    """Preview feedback data from Excel"""
    feedback_data = read_excel_feedback("feedback.xlsx")
    return jsonify({
        'count': len(feedback_data),
        'data': feedback_data[:10]
    })

@app.route('/user/ratings/<user_id>', methods=['GET'])
def get_user_ratings(user_id):
    """Get all ratings by a specific user"""
    feedback_data = read_excel_feedback("feedback.xlsx")
    user_ratings = [rating for rating in feedback_data if rating.get('user_id') == user_id]
    return jsonify({
        'user_id': user_id,
        'ratings_count': len(user_ratings),
        'ratings': user_ratings
    })

@app.route('/collaborative/retrain', methods=['POST'])
def retrain_collaborative():
    """Manually retrain collaborative filtering model"""
    try:
        success = initialize_collaborative_filtering()
        if success:
            return jsonify({"status": "success", "message": "Collaborative filtering model retrained successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to retrain collaborative filtering model"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("🎯 Final Collaborative Filtering Status:", "Enabled" if cf_initialized else "Disabled")
    if cf_initialized:
        print(f"👥 Users in system: {len(cf_system.user_ids)}")
        print(f"📊 Total items in matrix: {len(cf_system.item_titles)}")
    print("🎵 Supported moods:", SUPPORTED_MOODS)
    print("✅ CineTune backend with Collaborative Filtering running at http://127.0.0.1:5000")
    app.run(debug=True)