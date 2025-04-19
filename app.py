# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# Title
st.title("Titanic Survival Prediction")

# Load saved model and scaler
model = pickle.load(open("random_forest_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load data for evaluation/visualization only
df = pd.read_csv("train.csv")

# Preprocess data for evaluation
def preprocess_data(df):
    df = df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_scaled = scaler.transform(X)
    return X_scaled, y

X, y = preprocess_data(df)

# Accuracy
acc = accuracy_score(y, model.predict(X))
st.write(f"Model Accuracy: {acc:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y, model.predict(X))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Sidebar - User Input
def user_input():
    Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    Sex = st.selectbox("Sex", ['male', 'female'])
    Age = st.slider("Age", 0, 100, 25)
    SibSp = st.slider("Number of Siblings/Spouses Aboard", 0, 10, 0)
    Parch = st.slider("Number of Parents/Children Aboard", 0, 10, 0)
    Fare = st.slider("Fare", 0.0, 500.0, 50.0)
    Embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

    # Manual encoding to match training
    sex_mapped = 0 if Sex == 'male' else 1
    embarked_S = 1 if Embarked == 'S' else 0
    embarked_Q = 1 if Embarked == 'Q' else 0
    
    FamilySize = SibSp + Parch + 1
    IsAlone = 1 if FamilySize == 1 else 0

    data = {
        'Pclass': Pclass,
        'Sex': sex_mapped,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked_Q': embarked_Q,
        'Embarked_S': embarked_S,
        'FamilySize': FamilySize,
        'IsAlone': IsAlone
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Scale user input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)[0]
result = "Survived" if prediction == 1 else "Did Not Survive"

st.subheader("Prediction")
st.write(f"This passenger would have: **{result}**")
