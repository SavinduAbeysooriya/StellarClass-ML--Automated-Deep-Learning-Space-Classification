# StellarClass-ML: Automated Stellar Classification using Computational Intelligence

An end-to-end machine learning project to classify astronomical bodies (Stars, Galaxies, and Quasars) using spectral data from the Kaggle Playground Series (Season 6, Episode 6). This project is developed as part of the CIS6005 Computational Intelligence module at Cardiff Metropolitan University.

---

## 🌌 Project Overview
The objective of this project is to build a robust computational intelligence solution that automates the classification of cosmic objects. Using a large-scale dataset containing photometric features, the system trains, evaluates, and compares multiple models to optimize multi-class classification accuracy.

### Key Deliverables:
1. **Comprehensive EDA:** Extensive data visualizations exploring feature distributions and correlations.
2. **Machine Learning Pipeline:** Comparative analysis between **Ensemble Methods (Random Forests)** and **Artificial Neural Networks (ANN)**.
3. **Kaggle Leaderboard Submission:** Validated predictions submitted directly to the Kaggle platform.
4. **Interactive Web Application:** A functional Streamlit application that uses the trained model to predict stellar classes on user-inputted parameters.

---

## 🏗️ System Architecture
The system follows a standard production-grade ML pipeline:
1. **Data Acquisition & Preprocessing:** Handling missing data, feature scaling (StandardScaler), and encoding.
2. **Exploratory Data Analysis (EDA):** Feature distribution histograms, correlation heatmaps, and class balance analysis.
3. **Model Training:** Implementing Scikit-Learn (Random Forest) and TensorFlow/Keras (ANN).
4. **Evaluation:** Assessing models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
5. **Deployment:** Serving the best-performing model via a local Streamlit web server.

---

## 📂 Repository Structure
```text
├── data/                  # Local directory for datasets (Excluded from Git via .gitignore)
│   ├── train.csv
│   └── test.csv
├── notebooks/             # Jupyter Notebooks for experimentation
│   └── stellar_eda_and_modeling.ipynb
├── models/                # Saved model artifacts (.pkl or .h5)
├── app/                   # Streamlit Web Application source code
│   └── app.py
├── .gitignore             # Git ignore configurations
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation