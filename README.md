# ⚓ Titanic AI — Survival Prediction

> An end-to-end machine learning web application that predicts Titanic passenger survival using a tuned K-Nearest Neighbors (KNN) model.

Titanic AI transforms a traditional Titanic machine learning project into an interactive web application using **Python, Scikit-learn, Flask, HTML, CSS, and JavaScript**.

---

## 🚀 Project Overview

The application allows users to enter passenger information and receive a survival prediction from a trained machine learning model.

Instead of stopping at model training inside a Jupyter Notebook, this project takes the complete ML workflow into a working web application:

```text
Titanic Dataset
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Train / Test Split
       ↓
Feature Scaling
       ↓
Model Training
       ↓
Model Evaluation
       ↓
GridSearchCV
       ↓
Best KNN Model
       ↓
Model Serialization
       ↓
Flask Backend
       ↓
Interactive Web Interface

Features
🤖 Machine-learning powered survival prediction
🧠 K-Nearest Neighbors classifier
🎯 Hyperparameter tuning using GridSearchCV
📊 82.68% test accuracy
⚙️ Optimal K value of 11
📏 StandardScaler preprocessing
🌊 Titanic-inspired modern interface
✨ Animated glassmorphism UI
⚡ Real-time prediction through Flask API
🧑‍💻 Interactive passenger profile
📱 Responsive web design
🔄 Analyze multiple passengers without refreshing
🟢 Live AI system status indicator
📈 Model performance statistics
🧠 Machine-learning pipeline explanation

Machine Learning
Algorithm

The final model uses:

K-Nearest Neighbors (KNN)

Hyperparameter tuning was performed using:

GridSearchCV

The best-performing configuration was:

n_neighbors = 11

Five-fold cross-validation was used during hyperparameter selection.

Model Performance:

The final tuned KNN model achieved the following results on the test dataset:
| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 82.68% |
| Precision | 83.08% |
| Recall    | 72.97% |
| F1 Score  | 77.70% |

Best Configuration:
| Parameter        | Value               |
| ---------------- | ------------------- |
| Algorithm        | K-Nearest Neighbors |
| Optimal K        | 11                  |
| Cross Validation | 5-Fold              |
| Preprocessing    | StandardScaler      |

Features Used

The final model uses 18 processed features
Passenger Features:
Pclass
Sex
Age
SibSp
Parch
Fare

Embarkation Features:
Embarked_C
Embarked_Q
Embarked_S

Cabin Deck Features
CabinDeck_A
CabinDeck_B
CabinDeck_C
CabinDeck_D
CabinDeck_E
CabinDeck_F
CabinDeck_G
CabinDeck_T
CabinDeck_Unknown
The numerical features were standardized using StandardScaler.

Technologies Used:

Machine Learning:
Python
Pandas
NumPy
Scikit-learn
K-Nearest Neighbors
GridSearchCV
StandardScaler
Joblib

Backend:
Flask
Python
REST API

Frontend:
HTML5
CSS3
JavaScript

User Interface:

Titanic AI was designed as a modern AI-style application rather than a basic HTML form.

The interface includes:

⚓ Titanic AI branding
🌊 Animated particle background
✨ Glassmorphism design
🟢 AI system online indicator
👤 Passenger profile form
📊 Live passenger information preview
⚡ Animated prediction state
🎯 Interactive prediction result
🔄 Analyze-another-passenger functionality
📈 Model performance cards
🧠 ML pipeline explanation
📱 Responsive layout for smaller screens

How It Works:
The user enters information about a Titanic passenger, including:

Passenger class
Gender
Age
Ticket fare
Number of siblings/spouses
Number of parents/children
Port of embarkation
Cabin deck

The frontend sends the information to the Flask /predict endpoint.

The backend then:

1.Receives the passenger information.
2.Constructs the required feature set.
3.Applies the saved StandardScaler.
4.Passes the processed features to the trained KNN model.
5.Generates the survival prediction.
6.Returns the prediction to the frontend.
7.Displays the result using the interactive UI.

Project Structure:
Titanic_ML_Project/
│
├── app.py
│
├── titanic_knn_model.pkl
│
├── titanic_scaler.pkl
│
├── templates/
│   └── index.html
│
└── README.md

Important Files:
| File                    | Purpose                          |
| ----------------------- | -------------------------------- |
| `app.py`                | Flask backend and prediction API |
| `titanic_knn_model.pkl` | Saved trained KNN model          |
| `titanic_scaler.pkl`    | Saved StandardScaler             |
| `templates/index.html`  | Interactive frontend             |
| `README.md`             | Project documentation            |

Installation
1. Clone the Repository:
git clone https://github.com/YOUR_USERNAME/Titanic_ML_Project.git

2. Enter the Project Directory:
cd Titanic_ML_Project

3. Install Dependencies:
pip install flask pandas numpy scikit-learn joblib

Run the Application

Start the Flask server:
python app.py

The terminal should display:
* Running on http://127.0.0.1:5000

Open the application in your browser:
http://127.0.0.1:5000

Model Development

Several machine learning approaches were explored during development, including:

Logistic Regression
Decision Tree
Random Forest
K-Nearest Neighbors
Hyperparameter tuning
Cross-validation

The final KNN model was selected after evaluating the models and performing hyperparameter tuning with GridSearchCV.

The optimized model used:
K = 11

Model Serialization
After training, the final model was saved using Joblib:
joblib.dump(best_knn, "titanic_knn_model.pkl")
The scaler used during training was also saved:
joblib.dump(scaler, "titanic_scaler.pkl")

This allows the Flask application to load the trained model and preprocessing object without retraining every time the application starts.

Application Architecture:
                   TITANIC AI
                       │
                       ▼
              ┌─────────────────┐
              │  Web Interface  │
              │ HTML/CSS/JS     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Flask API     │
              │   /predict      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ StandardScaler  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   KNN Model     │
              │      K = 11     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Prediction    │
              │  Survived / Not │
              └─────────────────┘

Project Objective

The main objective of this project was to build a complete end-to-end machine learning application.
The project demonstrates how a machine learning model can move from:

Dataset
   ↓
Data Analysis
   ↓
Preprocessing
   ↓
Model Training
   ↓
Evaluation
   ↓
Hyperparameter Tuning
   ↓
Model Saving
   ↓
Flask API
   ↓
Web Application

Dataset:

The project uses the well-known Titanic passenger dataset.

The target variable is:

Survived

where:

0 → Did not survive
1 → Survived

The dataset contains information about passengers such as class, gender, age, family relationships, fare, embarkation port, and cabin information.


Future Improvements:
Possible future improvements include:

🌐 Deploy the application publicly
📊 Add prediction probabilities
📈 Add interactive data visualizations
🧠 Add explainable AI
🔍 Add model comparison dashboard
📊 Add passenger survival analytics
🐳 Containerize the application using Docker
🔐 Add production-grade API configuration
📱 Further improve accessibility and mobile experience

What This Project Demonstrates
This project demonstrates practical experience with:
Python
   +
Data Preprocessing
   +
Feature Engineering
   +
Machine Learning
   +
Model Evaluation
   +
Hyperparameter Tuning
   +
Model Serialization
   +
Flask API
   +
Frontend Development
   +
ML Application Deployment

Author

Tejas

Artificial Intelligence & Data Science

Project Highlights

A Titanic survival prediction model transformed into a complete interactive machine learning application.

Machine Learning
        +
Feature Engineering
        +
Hyperparameter Tuning
        +
Model Serialization
        +
Flask Backend
        +
Modern Frontend
        =
End-to-End ML Application

Disclaimer

This project is an educational machine learning application based on historical Titanic passenger data.

The predictions are generated by a statistical machine learning model and should not be interpreted as factual historical conclusions.
