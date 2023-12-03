# 66552024_MovieSuccess_FinalProject
A Repository Holding a Movie Success Prediction Model for the Intro to Ai's Final Summative Group Project 


## Overview
This project introduces a Python-based movie success prediction system, leveraging Random Forest, XGBoost, and Multilayer Perceptron models. Trained on the Kaggle Movies MetaData Dataset, the ensemble model combines predictions based on diverse attributes like budget, popularity,vote ratings, number of voters,revenue and genres. Targeted at investors, filmmakers, and viewers, the system aids in risk assessment, decision-making, and movie recommendations. Success is defined by a profit margin exceeding 50%. This project aims to revolutionize the film industry with a concise and powerful approach to predicting movie success.



The Steps Enacted Include:

1.Data Preparation & Feature Extraction: filling missing values, visualization,cleaning data,encoding etc.

2. Feature Engineering and Scaling

3. Model Creation and Training: by splitting data and creating ensemble model combining Random Forest, XGBoost, and Multilayer Perceptron Model for predicting movie success.
   
4. Model Performance Measurement and Fine-Tuning(accuracy, precision, recall, etc.).
   
5. Model Deployment on a Web Page using streamlit that incorporates the trained movie success prediction model.
   
6. Video Demonstration demonstrating the application's functionality.  [Watch the video demonstration here](https://youtu.be/sPpoWTDliCU))



## Repository Contents
Database Used : Kaggle's The Movies Dataset specifically file 'movies_metadata.csv'

Google Colab Notebook named 66552024_MovieSuccessPrediction.ipynb

4 Saved Models & Scaler

app.py :Consists of the web-based application code for deploying the model

Video_Demonstration

README.md :This file

Requirements.txt file



## How To host It (either on a local server or on the cloud)
I will use my experience to help you understand how to host it:

Initially, the system was hosted locally using a combination of Python tools and libraries. The preprocessing scaler and trained models were saved, and the necessary dependencies were documented in the requirements file. To ensure a seamless local deployment, all components were organized into a single folder, and an app.py script using Streamlit was created. This script served as the entry point for the application. The deployment was executed within the Visual Studio Code (VSCode) environment, configuring the necessary software and dependencies. The local server setup involved running the Streamlit app on a local server accessible within the network, allowing for interface design, testing, and development.

For the final deployment, the project transitioned to cloud hosting. Using Streamlit sharing, the application was deployed on the cloud for increased accessibility and scalability. The cloud deployment facilitated wider accessibility, enabling users to interact with the movie success prediction system from any location. This transition from local to cloud hosting was a strategic move for the final deployment of the system, leveraging the advantages of cloud infrastructure for improved performance and availability.


## Model Deployment
Model was deployed to `Streamlit` using the  for prediction.


[Link to Deployed App on Streamlit](https://66552024moviesuccessfinalproject-xwyrqgxk7xhpme33yg2kmk.streamlit.app/)
