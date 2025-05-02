import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page Streamlit
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #F0F2F6;
        color: #262730;
    }
    .stButton>button {
        background-color: #6C63FF;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #8A84FF;
        transform: scale(1.05);
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #E0E0E0;
    }
    .movie-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #E0E0E0;
        transition: all 0.3s ease;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .movie-card h3 {
        color: #262730;
        margin-bottom: 10px;
    }
    .movie-card p {
        color: #4B4B4B;
        margin: 5px 0;
    }
    .title {
        font-size: 2.5em;
        font-weight: bold;
        color: #6C63FF;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .subtitle {
        font-size: 1.5em;
        color: #262730;
        margin-bottom: 20px;
        border-bottom: 2px solid #6C63FF;
        padding-bottom: 10px;
    }
    .stSidebar {
        background-color: white;
    }
    .stSidebar .sidebar-content {
        background-color: white;
    }
    .stInfo {
        background-color: white;
        border: 1px solid #E0E0E0;
    }
    .stNumberInput>div>div>input {
        background-color: white;
        color: #262730;
        border: 1px solid #E0E0E0;
    }
    .stNumberInput>div>div>input:focus {
        border-color: #6C63FF;
    }
    .stSpinner>div {
        border-color: #6C63FF;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #6C63FF;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: #4B4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="title">🎬 Movie Recommender</div>', unsafe_allow_html=True)

# Fonction pour charger les données
@st.cache_data
def load_data():
    # Charger les données MovieLens
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    movies = pd.read_csv('ml-latest-small/movies.csv')
    return ratings, movies

# Fonction pour entraîner le modèle
@st.cache_resource
def train_model(ratings):
    # Créer un reader pour Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    # Diviser les données en ensembles d'entraînement et de test
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Entraîner le modèle SVD
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    
    return model, testset

# Fonction pour obtenir les recommandations
def get_recommendations(model, user_id, movies, n_recommendations=5):
    # Obtenir tous les films
    all_movies = movies['movieId'].unique()
    
    # Prédire les notes pour tous les films
    predictions = []
    for movie_id in all_movies:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))
    
    # Trier les prédictions et obtenir les meilleures recommandations
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n_recommendations]
    
    # Obtenir les détails des films recommandés
    recommended_movies = movies[movies['movieId'].isin([x[0] for x in top_n])].copy()
    recommended_movies['predicted_rating'] = [x[1] for x in top_n]
    
    return recommended_movies

def main():
    # Charger les données
    try:
        with st.spinner("Chargement des données..."):
            ratings, movies = load_data()
        
        # Sidebar pour les statistiques
        with st.sidebar:
            st.markdown('<div class="subtitle">📊 Statistiques</div>', unsafe_allow_html=True)
            st.metric("👥 Utilisateurs", f"{ratings['userId'].nunique():,}")
            st.metric("🎬 Films", f"{movies['movieId'].nunique():,}")
            st.metric("⭐ Évaluations", f"{len(ratings):,}")
            
            st.markdown("---")
            st.markdown("### ℹ️ À propos")
            st.info("Ce système utilise l'algorithme SVD pour recommander des films basés sur vos préférences et celles des autres utilisateurs.")
        
        # Section principale
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="subtitle">🎯 Obtenir des recommandations</div>', unsafe_allow_html=True)
            user_id = st.number_input(
                "Entrez votre ID utilisateur",
                min_value=1,
                max_value=610,
                value=1,
                help="Choisissez un ID entre 1 et 610"
            )
            
            if st.button("✨ Obtenir des recommandations", use_container_width=True):
                with st.spinner("Entraînement du modèle en cours..."):
                    model, testset = train_model(ratings)
                
                recommendations = get_recommendations(model, user_id, movies)
                
                st.markdown('<div class="subtitle">🎬 Films recommandés pour vous</div>', unsafe_allow_html=True)
                for _, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"""
                            <div class="movie-card">
                                <h3>{row['title']}</h3>
                                <p>Note prédite: ⭐ {row['predicted_rating']:.2f}/5.0</p>
                            </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="subtitle">📈 Top Films</div>', unsafe_allow_html=True)
            top_movies = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
            top_movies = top_movies.merge(movies, on='movieId')
            top_movies = top_movies[top_movies['count'] > 50].sort_values('mean', ascending=False).head(5)
            
            for _, row in top_movies.iterrows():
                with st.container():
                    st.markdown(f"""
                        <div class="movie-card">
                            <h3>{row['title']}</h3>
                            <p>Note moyenne: ⭐ {row['mean']:.2f}/5.0</p>
                            <p>Nombre d'évaluations: {row['count']}</p>
                        </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        st.info("Veuillez télécharger le dataset MovieLens et le placer dans le dossier 'ml-latest-small'")

if __name__ == "__main__":
    main() 