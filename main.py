import streamlit as st
import joblib
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .movie-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    .movie-title {
        font-size: 1rem;
        font-weight: bold;
        margin-top: 0.5rem;
        color: #333;
        text-align: center;
    }
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# OMDb API key (replace with your own)
OMDB_API_KEY = "1a71d6fd"

@st.cache_data
def load_data():
    """Load the preprocessed data and models"""
    try:
        df = joblib.load("df.pkl")
        vectors = joblib.load("vectors.pkl")
        model = joblib.load("model.pkl")
        return df, vectors, model
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure df.pkl, vectors.pkl, and model.pkl are in the same directory.")
        return None, None, None

def get_movie_poster(movie_id):
    """Fetch movie poster from OMDb API"""
    try:
        url = f'http://www.omdbapi.com/?i={movie_id}&apikey={OMDB_API_KEY}'
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data.get('Response') == 'True':
            poster_url = data.get('Poster', '')
            if poster_url and poster_url != 'N/A':
                return poster_url
    except:
        pass
    
    # Return placeholder image if poster not found
    return "https://via.placeholder.com/300x450/667eea/ffffff?text=No+Poster"

def get_movie_details(movie_id):
    """Fetch additional movie details from OMDb API"""
    try:
        url = f'http://www.omdbapi.com/?i={movie_id}&apikey={OMDB_API_KEY}'
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data.get('Response') == 'True':
            return {
                'year': data.get('Year', 'N/A'),
                'rated': data.get('Rated', 'N/A'),
                'runtime': data.get('Runtime', 'N/A'),
                'imdb_rating': data.get('imdbRating', 'N/A'),
                'plot': data.get('Plot', 'No plot available')
            }
    except:
        pass
    
    return None

def recommend_movies(movie_name, df, vectors, model, n_recommendations=5):
    """Generate movie recommendations"""
    try:
        # Find the movie index
        index = df[df['name'] == movie_name].index[0]
        test_vector = vectors[index]
        
        # Get similar movies
        scores, indexes = model.kneighbors([test_vector], n_neighbors=n_recommendations + 1)
        
        # Exclude the first result (the movie itself)
        recommendations = []
        for i, idx in enumerate(indexes[0][1:]):
            movie_data = df.iloc[idx]
            similarity_score = 1 - scores[0][i + 1]  # Convert distance to similarity
            recommendations.append({
                'name': movie_data['name'],
                'movie_id': movie_data['movie_id'],
                'similarity': similarity_score
            })
        
        return recommendations
    except IndexError:
        return None

def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 2rem;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    
    # Load data
    df, vectors, model = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2503/2503508.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Choose a page:",
            ["🏠 Home", "🔍 Find Recommendations", "📊 Dataset Explorer", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Movies", len(df))
        with col2:
            st.metric("Features", vectors.shape[1])
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "🔍 Find Recommendations":
        show_recommendations_page(df, vectors, model)
    elif page == "📊 Dataset Explorer":
        show_explorer_page(df)
    else:
        show_about_page()

def show_home_page(df):
    """Display home page with featured movies"""
    st.markdown("<div style='background: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    st.markdown("## Welcome to the Movie Recommendation System")
    st.markdown("""
    This intelligent system uses **Machine Learning** to recommend movies based on content similarity.
    
    ### Features:
    - 🎯 **Content-Based Filtering**: Recommendations based on movie descriptions, genres, directors, and languages
    - 🤖 **ML-Powered**: Uses TF-IDF vectorization and Nearest Neighbors algorithm
    - 🎨 **Visual Interface**: Browse movies with posters and detailed information
    - 📈 **Similarity Scores**: See how similar recommended movies are to your selection
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Featured movies
    st.markdown("### 🌟 Featured Movies")
    featured_indices = df.sample(6).index
    
    cols = st.columns(6)
    for idx, col_idx in enumerate(featured_indices):
        with cols[idx]:
            movie = df.iloc[col_idx]
            poster_url = get_movie_poster(movie['movie_id'])
            st.image(poster_url, use_container_width=True)
            st.markdown(f"<div class='movie-title'>{movie['name'][:30]}...</div>", unsafe_allow_html=True)

def show_recommendations_page(df, vectors, model):
    """Display recommendations page"""
    st.markdown("<div style='background: white; padding: 2rem; border-radius: 10px;'>", unsafe_allow_html=True)
    st.markdown("## 🔍 Find Similar Movies")
    st.markdown("Select a movie to get personalized recommendations based on content similarity.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Movie selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_movie = st.selectbox(
            "Choose a movie:",
            options=sorted(df['name'].unique()),
            index=0
        )
    
    with col2:
        n_recommendations = st.slider(
            "Number of recommendations:",
            min_value=3,
            max_value=10,
            value=5
        )
    
    if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Finding similar movies..."):
            recommendations = recommend_movies(selected_movie, df, vectors, model, n_recommendations)
            
            if recommendations:
                # Display selected movie
                st.markdown("---")
                st.markdown("### 📽️ Selected Movie")
                
                selected_movie_data = df[df['name'] == selected_movie].iloc[0]
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    poster_url = get_movie_poster(selected_movie_data['movie_id'])
                    st.image(poster_url, use_container_width=True)
                
                with col2:
                    st.markdown(f"### {selected_movie}")
                    details = get_movie_details(selected_movie_data['movie_id'])
                    if details:
                        st.markdown(f"**Year:** {details['year']} | **Rating:** {details['rated']} | **Runtime:** {details['runtime']} | **IMDb:** ⭐ {details['imdb_rating']}")
                        st.markdown(f"**Plot:** {details['plot']}")
                
                # Display recommendations
                st.markdown("---")
                st.markdown("### 🎯 Recommended Movies")
                
                cols = st.columns(min(5, n_recommendations))
                for idx, rec in enumerate(recommendations):
                    with cols[idx % 5]:
                        poster_url = get_movie_poster(rec['movie_id'])
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"<div class='movie-title'>{rec['name'][:30]}</div>", unsafe_allow_html=True)
                        st.progress(rec['similarity'])
                        st.caption(f"Match: {rec['similarity']*100:.1f}%")
                
                # Detailed list
                st.markdown("### 📋 Detailed Recommendations")
                for idx, rec in enumerate(recommendations, 1):
                    with st.expander(f"{idx}. {rec['name']} - {rec['similarity']*100:.1f}% match"):
                        details = get_movie_details(rec['movie_id'])
                        if details:
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.markdown(f"**Year:** {details['year']}")
                                st.markdown(f"**Rating:** {details['rated']}")
                                st.markdown(f"**Runtime:** {details['runtime']}")
                                st.markdown(f"**IMDb:** ⭐ {details['imdb_rating']}")
                            with col2:
                                st.markdown(f"**Plot:** {details['plot']}")
            else:
                st.error("Movie not found. Please try another selection.")

def show_explorer_page(df):
    """Display dataset explorer page"""
    st.markdown("<div style='background: white; padding: 2rem; border-radius: 10px;'>", unsafe_allow_html=True)
    st.markdown("## 📊 Dataset Explorer")
    st.markdown(f"Explore the dataset of **{len(df)}** movies used in the recommendation system.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Search functionality
    search_term = st.text_input("🔍 Search movies:", placeholder="Enter movie name...")
    
    if search_term:
        filtered_df = df[df['name'].str.contains(search_term, case=False, na=False)]
    else:
        filtered_df = df
    
    # Display results
    st.markdown(f"### Showing {len(filtered_df)} movies")
    
    # Pagination
    items_per_page = 20
    total_pages = (len(filtered_df) - 1) // items_per_page + 1
    page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
    
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    
    # Display movies in grid
    display_df = filtered_df.iloc[start_idx:end_idx]
    
    for i in range(0, len(display_df), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(display_df):
                movie = display_df.iloc[i + j]
                with cols[j]:
                    poster_url = get_movie_poster(movie['movie_id'])
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"<div class='movie-title'>{movie['name'][:30]}</div>", unsafe_allow_html=True)

def show_about_page():
    """Display about page"""
    st.markdown("<div style='background: white; padding: 2rem; border-radius: 10px;'>", unsafe_allow_html=True)
    st.markdown("## ℹ️ About This System")
    
    st.markdown("""
    ### 🎯 How It Works
    
    This movie recommendation system uses **Content-Based Filtering** with machine learning to suggest similar movies.
    
    #### Technical Details:
    
    1. **Data Processing**
       - Dataset: 1,752 movies with descriptions, genres, directors, and languages
       - Features combined into a single text representation
    
    2. **Text Vectorization**
       - **TF-IDF (Term Frequency-Inverse Document Frequency)** converts text to numerical vectors
       - Captures importance of words across the dataset
    
    3. **Similarity Computation**
       - **Nearest Neighbors** algorithm with cosine similarity
       - Finds movies with similar content profiles
    
    4. **Recommendation Generation**
       - Returns top-N most similar movies
       - Provides similarity scores for transparency
    
    ### 🛠️ Technology Stack
    
    - **Python**: Core programming language
    - **Streamlit**: Web application framework
    - **scikit-learn**: Machine learning algorithms
    - **Pandas**: Data manipulation
    - **OMDb API**: Movie posters and metadata
    
    ### 📚 Use Cases
    
    - **Streaming Platforms**: Suggest content to users
    - **Content Discovery**: Help users find new movies
    - **Market Research**: Analyze content similarities
    - **Entertainment Apps**: Enhance user engagement
    
    ### 🔮 Future Enhancements
    
    - Hybrid recommendations (collaborative + content-based)
    - User rating integration
    - Real-time learning from user preferences
    - Multi-language support
    - Advanced filtering options
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='background: white; padding: 1rem; border-radius: 10px; text-align: center;'>", unsafe_allow_html=True)
    st.markdown("### 📧 Contact & Support")
    st.markdown("Built with ❤️ using Machine Learning | © 2024")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()