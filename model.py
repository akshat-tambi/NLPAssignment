import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

# stopwords from NLTK
nltk.download('stopwords')
os.system('clear')

# data load
url = "https://github.com/akshat-tambi/NLPAssignment/raw/refs/heads/main/spam.csv"
data = pd.read_csv(url, encoding='ISO-8859-1')

# now we parse the data
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# text preprocessing
def preprocess_text(text):
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower() 
    english_stopwords = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in english_stopwords]) 
    return text

# preprocess the dataset using our preprocessing function
data['message'] = data['message'].apply(preprocess_text)

# ham => 0 and spam => 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# an 80/10/10 split of dataset into training and dev and test
X_train, X_temp, y_train, y_temp = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42, stratify=data['label']) 
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)  

# we vectorize the text through TF-IDF [bi and tri-grams]
vectorizer = TfidfVectorizer(ngram_range=(2, 3), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_dev_vec = vectorizer.transform(X_dev)
X_test_vec = vectorizer.transform(X_test)



# we implement the naive bayes model along with hyperparameter tuning {alpha}
# alpha range based on the last best alpha found

# variables for tuning
low_accuracy_threshold = 0.971
max_iterations = 100  
iteration = 0
best_accuracy = 0.0
patience = 50
no_improvement_count = 0
nb_model = MultinomialNB()

while iteration < max_iterations:
    
    if best_accuracy > 0:
        alpha_range = np.arange(max(0.1, grid.best_params_['alpha'] - 0.3), 
                                min(2.0, grid.best_params_['alpha'] + 0.3), 
                                0.05)  
    else:
        alpha_range = np.arange(0.1, 2.1, 0.1)  

    param_grid = {'alpha': alpha_range} 
    grid = GridSearchCV(nb_model, param_grid, cv=5, verbose=0)
    grid.fit(X_train_vec, y_train)

    nb_dev_pred = grid.predict(X_dev_vec)
    nb_dev_accuracy = accuracy_score(y_dev, nb_dev_pred)

    # print("Best alpha value found by GridSearch:", grid.best_params_)
    # print(f"Naive Bayes - Correct Predictions on Development Set: {round(nb_dev_accuracy * 100, 2)}%")
    # print("Naive Bayes Report on Development Set:\n", classification_report(y_dev, nb_dev_pred))

    if nb_dev_accuracy > best_accuracy:
        best_accuracy = nb_dev_accuracy
        no_improvement_count = 0
        if nb_dev_accuracy >= low_accuracy_threshold:
            # print("Achieved satisfactory accuracy, stopping tuning.")
            break
    else:
        # print("Accuracy did not improve.")
        no_improvement_count += 1

    if no_improvement_count >= patience:
        # print("No improvement for several iterations, stopping tuning.")
        break
    
    iteration += 1

print(f"Iterations completed: {iteration}")

# test set
nb_test_pred = grid.predict(X_test_vec)
nb_test_accuracy = accuracy_score(y_test, nb_test_pred)
print(f"Naive Bayes - Correct Predictions on Test Set: {round(nb_test_accuracy * 100, 2)}%")
# print("Naive Bayes Report on Test Set:\n", classification_report(y_test, nb_test_pred))

import joblib
joblib.dump(grid, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# user function input and output
def predict_message(message):
    message = preprocess_text(message) 
    message_vec = vectorizer.transform([message])  
    prediction = grid.predict(message_vec)
    return "Spam" if prediction[0] == 1 else "Ham"

while True:
    user_input = input("Enter a message to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    result = predict_message(user_input)
    print(f"The message is classified as: {result}")