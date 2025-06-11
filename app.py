import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load full dataset from Excel
@st.cache_data
def load_data():
    df = pd.read_excel("Oak Gall Formers Database  3.0.xlsx", sheet_name="Michigan Full List")
    df.columns = df.columns.str.strip()
    df = df[['Plant Species', 'Emergence time', 'Tissue', 'Form', 'Alternative generation?', 'Insect Species']]
    df = df.dropna(subset=['Insect Species'])
    return df

data = load_data()

# Define features and target
y = data['Insect Species']
X = data[['Plant Species', 'Emergence time', 'Tissue', 'Form', 'Alternative generation?']]
categorical_features = X.columns.tolist()

# Preprocessor and model pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, probability=True))
])

# Fit the model
model.fit(X, y)

