from setuptools import setup

setup(
    name="movie-recommender",
    version="0.1",
    install_requires=[
        "numpy==1.23.5",
        "pandas==1.5.3",
        "scikit-learn==1.2.2",
        "streamlit==1.24.0",
        "matplotlib==3.7.1",
        "seaborn==0.12.2",
        "scikit-surprise==1.0.6",
    ],
    python_requires=">=3.9",
) 