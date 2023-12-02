# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import streamlit as st
import pickle




from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

# Set secondary color to purple
st.set_page_config(
    page_title="Movie Success Prediction 🎬💸",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to scale user input
def scale_input(user_input):

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    user_input_df = pd.DataFrame([user_input], columns=feature_names)
    scaled_input = scaler.transform(user_input_df)
    return  pd.DataFrame(scaled_input, columns=user_input_df.columns)



# Load trained model 
with open('ensemble_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Feature names
feature_names = ['budget', 'popularity', 'revenue', 'vote_average', 'vote_count', 'genres']

st.title('Movie Success Prediction 🎬💸')

# User input fields
st.sidebar.header('Movie Features')

# Mapping of genre names to their corresponding numbers
genre_mapping = {
    1: 'Drama',
    2: 'Comedy',
    3: 'Action',
    4: 'Adventure',
    5: 'Horror',
    6: 'Crime',
    7: 'Thriller',
    8: 'Animation',
    9: 'Fantasy',
    10: 'Romance',
    11: 'Science Fiction',
    12: 'Mystery',
    13: 'Family',
    14: 'Documentary',
    15: 'War',
    16: 'Music',
    17: 'Western',
    18: 'History',
    19: 'Foreign',
    20: 'TV Movie'
}
def user_input_features():
    budget = st.sidebar.number_input('Enter Budget (USD)', min_value=0, max_value=int(1e9), value=int(1e9))
    popularity = st.sidebar.number_input('Movie Popularity', min_value=0, max_value=int(1000), value=int(1e3))
    revenue = st.sidebar.number_input('Enter Revenue Grossed (USD)', min_value=0, max_value=int(1e9), value=int(1e9))
    vote_average = st.sidebar.slider('Voting Average Out of 10', 0, 5, 10)
    vote_count = st.sidebar.slider('Vote Count for Movie', 1, 10000, 20000)


    genres = st.sidebar.selectbox('Genres', list(genre_mapping.values()), index=0)
    genre_number = next(key for key, value in genre_mapping.items() if value == genres)
    data = {
        'budget': budget,
        'popularity': popularity,
        'revenue': revenue,
        'vote_average': vote_average,
        'vote_count': vote_count,
        'genres': genre_number
    }
    return data

input_data = user_input_features()

# Get predictions from model
st.subheader('Lets Predict The Success of Your Movie Today!👩🏽‍💻🔄🥁')
scaled_input = scale_input(input_data)
prediction = model.predict(scaled_input)
if prediction[0] ==0:

    st.write("The Movie was a Successs!!🎉🎟️ ")
else:
    st.write("Oh No!😔 This Movie wasn't Successful. Don't Bother to Watch or Invest!🚫")

# Explain model's prediction
if st.button('Prediction Explanation'):
    st.write("In this App, we are using a simple ensemble model that averages the predictions of 3 different models: Random Forest, XGBoost and Multilayer Perceptron Model in order to predict whether a Movie is Successful or Not.")
    st.write("The model was trained on the Movies MetaData Dataset by Kaggle, which contains data on thousands of movies. The dataset contains information on a movie's attributes such as its budget, popularity score, revenue, voting average rating out of 10, the voting count and its genre. The model was trained to predict the movie's success based on these attributes but most specifically its profit based on its initial budget and revenue generated.If the profit is more than 50% then the movie is considered successful, otherwise its unsuccessful.")
    st.write("This is a demo project and doesn't use any advanced model explanation techniques. Use with caution.")


