# Spam Message Classifier

This project is a simple NLP-based spam detection system that classifies messages as **spam** or **not spam** (ham) using a Naive Bayes classifier. It includes a **React** frontend and a **Flask** backend. The model utilizes **TF-IDF vectorization** and **Multinomial Naive Bayes**, with **hyperparameter tuning** done using `GridSearchCV`.

## Features

- **Naive Bayes Classifier**: Uses the Multinomial Naive Bayes algorithm for classification.
- **TF-IDF Vectorization**: Vectorizes text data with bigram and trigram features.
- **Hyperparameter Tuning**: Optimizes the model's `alpha` parameter using `GridSearchCV`.
- **Flask API**: The backend serves the trained model and responds to predictions.
- **React Frontend**: A simple interface to input a message and get its classification.
- **Cross-Origin Resource Sharing (CORS)**: Enabled to allow communication between the frontend and backend.

## Technologies Used

- **Machine Learning & NLP**: 
  - `pandas`, `nltk`, `scikit-learn` for data handling, text preprocessing, vectorization, and model training.
  - `joblib` for saving and loading the model and vectorizer.
  
- **Backend**: 
  - `Flask` for serving the model.
  - `flask-cors` for handling CORS issues between the React app and the Flask API.
  
- **Frontend**: 
  - **React** (with **Vite**) for the user interface.

## How it Works

1. **Text Preprocessing**: 
   - Non-alphabetic characters and digits are removed.
   - Text is converted to lowercase.
   - Stop words (common words like "the", "and") are removed using the `nltk` stopwords list.

2. **TF-IDF Vectorization**:
   - The text data is transformed into vectors using the **TF-IDF (Term Frequency-Inverse Document Frequency)** method with **bi-grams** and **tri-grams**.

3. **Naive Bayes Classification**:
   - A **Multinomial Naive Bayes** classifier is trained on the vectorized data.
   - **GridSearchCV** is used for hyperparameter tuning (specifically optimizing the `alpha` parameter).

4. **User Interaction**:
   - The user inputs a message in the frontend (React app).
   - The message is sent to the Flask API, which preprocesses the message and returns a prediction (either `Spam` or `Not Spam`).
