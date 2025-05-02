# Movie Recommendation System

A sophisticated movie recommendation system built with Python, utilizing collaborative filtering and matrix factorization techniques. This project demonstrates the implementation of a machine learning-based recommendation engine using the MovieLens dataset.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Oumllack/Movie-Recommandation-and-Collaborative-Filtering.git)

## 🎯 Features

- **Personalized Movie Recommendations**: Get movie suggestions based on user preferences
- **Real-time Model Training**: SVD (Singular Value Decomposition) algorithm implementation
- **Modern Web Interface**: Built with Streamlit for an intuitive user experience
- **Performance Metrics**: View system statistics and top-rated movies
- **Responsive Design**: Beautiful and user-friendly interface

## 🛠️ Technical Stack

- **Python 3.x**
- **Libraries**:
  - `surprise`: For implementing the SVD algorithm
  - `pandas`: For data manipulation
  - `numpy`: For numerical computations
  - `streamlit`: For the web interface
  - `matplotlib` & `seaborn`: For data visualization

## 📊 Architecture

### Machine Learning Components

1. **Data Processing**
   - Uses MovieLens dataset (ml-latest-small)
   - Handles user ratings and movie metadata
   - Implements data preprocessing and transformation

2. **Model Architecture**
   - SVD (Singular Value Decomposition) implementation
   - Matrix factorization approach
   - Parameters:
     - n_factors: 100 (latent features)
     - n_epochs: 20 (training iterations)
     - learning rate: 0.005
     - regularization: 0.02

3. **Recommendation Engine**
   - Collaborative filtering approach
   - Real-time prediction generation
   - Top-N recommendations system

### Web Interface

1. **Main Components**
   - User input section
   - Recommendation display
   - Statistics dashboard
   - Top movies section

2. **UI Features**
   - Modern dark theme
   - Responsive design
   - Interactive elements
   - Loading indicators

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Oumllack/Movie-Recommandation-and-Collaborative-Filtering.git
cd Movie-Recommandation-and-Collaborative-Filtering
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the MovieLens dataset:
   - Download the "ml-latest-small" dataset from [MovieLens](https://grouplens.org/datasets/movielens/)
   - Extract the files into a `ml-latest-small` directory in the project root

## 💻 Usage

1. Start the application:
```bash
./venv/bin/streamlit run movie_recommender.py
```

2. Access the web interface at `http://localhost:8501`

3. Enter a user ID (1-610) to get personalized movie recommendations

## ⚠️ Known Issues and Solutions

1. **ModuleNotFoundError: No module named 'surprise'**
   - Solution: Make sure you're using the virtual environment and have installed all dependencies:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **SettingWithCopyWarning in Pandas**
   - This is a warning, not an error
   - The application will still work correctly
   - To fix, use `.loc` for DataFrame modifications

3. **Watchdog Performance Warning**
   - For better performance, install Watchdog:
   ```bash
   xcode-select --install  # On macOS
   pip install watchdog
   ```

## 📈 Performance

The system uses the following metrics for evaluation:
- RMSE (Root Mean Square Error) for model accuracy
- Coverage of recommendations
- Diversity of suggested movies

## 🔧 Configuration

The system can be configured by modifying the following parameters in `movie_recommender.py`:
- Number of recommendations
- Model parameters
- UI customization options

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MovieLens dataset provided by GroupLens Research
- Surprise library for the SVD implementation
- Streamlit for the web interface framework 