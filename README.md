# 🛡️ Network Intrusion & Anomaly Detection System

An unsupervised machine learning project for detecting anomalous network traffic using the CICIDS2017 dataset.

## 📌 Project Overview

Network intrusion detection is an important cybersecurity task used to identify suspicious or abnormal network activity.

This project explores multiple machine learning and deep learning techniques to detect anomalies in network traffic.

The project uses:

- PCA for dimensionality reduction and visualization
- K-Means for clustering
- DBSCAN for density-based clustering
- Isolation Forest for anomaly detection
- Autoencoder for reconstruction-based anomaly detection

## 📊 Dataset

This project uses the **CICIDS2017** dataset created by the Canadian Institute for Cybersecurity at the University of New Brunswick.

The dataset contains network traffic representing benign activity and several types of attacks.

The original dataset is not included in this repository because of its large size.

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

## 🧠 Machine Learning Pipeline

```text
CICIDS2017 Dataset
        ↓
Data Cleaning
        ↓
Duplicate Removal
        ↓
Missing & Infinite Value Handling
        ↓
Feature Selection
        ↓
Standard Scaling
        ↓
PCA
        ↓
Clustering
 ┌──────┼─────────┐
 ↓      ↓         ↓
K-Means DBSCAN   Visualization
        ↓
Anomaly Detection
 ┌───────────────┐
 ↓               ↓
Isolation Forest Autoencoder
        ↓
Model Evaluation
        ↓
Streamlit Dashboard

📈 Model Results

The models were evaluated using a sampled dataset containing normal and attack traffic.

Model	Accuracy	Precision	Recall	F1-Score
Isolation Forest	82.78%	47.80%	29.55%	36.52%
Autoencoder	88.74%	70.05%	57.34%	63.06%

The Autoencoder was evaluated using reconstruction error with a threshold calculated from normal training traffic.

🖥️ Streamlit Application

The project includes an interactive Streamlit dashboard that allows users to upload network traffic data and obtain anomaly predictions.

The dashboard provides:

Traffic overview
Dataset preview
Feature validation
Isolation Forest predictions
Autoencoder predictions
Model comparison
Consensus analysis
Flow-level anomaly predictions
📁 Project Structure
Network_Intrusion_Anomaly_Detection/
│
├── App/
│   └── app.py
│
├── Notebooks/
│   └── anomally_detection.ipynb
│
├── Models/
│   └── Model files are excluded from GitHub
│
├── Data/
│   └── Dataset files are excluded from GitHub
│
├── .gitignore
└── README.md

🚀 How to Run
1. Clone the repository
git clone https://github.com/Rakesheralla/Network_Intrusion_Anomaly_Detection.git
2. Navigate to the project
cd Network_Intrusion_Anomaly_Detection
3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow streamlit joblib
4. Run the Streamlit application
cd App
streamlit run app.py
🎯 Learning Objectives

This project demonstrates practical understanding of:

Unsupervised learning
Clustering
Dimensionality reduction
Anomaly detection
Deep learning
Reconstruction error
Model evaluation
Data preprocessing
Streamlit deployment
Git and GitHub workflow
👨‍💻 Author

Rakesh Eralla

GitHub: https://github.com/Rakesheralla


