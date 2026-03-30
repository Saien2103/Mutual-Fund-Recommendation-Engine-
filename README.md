# Mutual-Fund-Recommendation-Engine-
# 🎯 Mutual Fund Recommendation System

## 💡 Mission: Financial Literacy for Youngsters
Many young adults find the world of investing intimidating due to the overwhelming number of choices and complex jargon. 

I built this project to:
* **Simplify the Entry Point:** Help beginners understand which funds match their specific age and income.
* **Encourage Data-Driven Decisions:** Move away from "guesswork" and show how risk and duration impact investment choices.
* **Bridge the Knowledge Gap:** Provide a starting point for those who want to start building wealth but don't know where to begin.
  
A Machine Learning-powered web application that recommends the top 3 mutual funds for a user based on their financial profile (Age, Income, Risk Appetite, etc.). Built with **Python**, **Scikit-Learn**, and **Streamlit**.

## 🚀 Features
* **Machine Learning Backend:** Uses a Random Forest Classifier to analyze historical fund data.
* **Top-3 Recommendations:** Provides multiple options to the user using probability-based scoring.
* **Interactive UI:** A clean, easy-to-use interface built with Streamlit.
* **Data Cleaning:** Automatically handles currency symbols and percentage signs for processing.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **ML Library:** Scikit-Learn
* **Data Handling:** Pandas, Numpy
* **Web Framework:** Streamlit
* **Model Storage:** Joblib

## 📁 Project Structure
* `app.py`: The main Streamlit web application code.
* `Mutual_Funds_500.csv`: The dataset used for training and feature reference.
* `fund_model.pkl`: The trained Random Forest model.
* `encoders.pkl`: Saved LabelEncoders for processing user input.
* `requirements.txt`: List of required Python libraries.
