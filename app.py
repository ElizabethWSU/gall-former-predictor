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
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the model
model.fit(X, y)

# Streamlit UI
st.title("Gall Former Predictor")
st.write("Enter gall traits to predict the most likely gall-forming wasp species.")

plant = st.selectbox("Plant Species", options=data['Plant Species'].dropna().unique())
emergence = st.selectbox("Emergence Time", options=data['Emergence time'].dropna().unique())
tissue = st.selectbox("Tissue", options=data['Tissue'].dropna().unique())
form = st.selectbox("Form", options=data['Form'].dropna().unique())
generation = st.selectbox("Alternative Generation?", options=data['Alternative generation?'].dropna().unique())

input_df = pd.DataFrame([{
    'Plant Species': plant,
    'Emergence time': emergence,
    'Tissue': tissue,
    'Form': form,
    'Alternative generation?': generation
}])

if st.button("Predict Gall Former"):
    pred = model.predict(input_df)
    st.subheader("Predicted Gall-Forming Species:")
    st.write(pred[0])
