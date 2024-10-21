from flask import Flask, request, jsonify
import joblib
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the saved model and vectorizer
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load stopwords from the local folder
stopwords_path = ('english_stopwords')
with open(stopwords_path, 'r') as f:
    english_stopwords = set(f.read().splitlines())

# Function to preprocess input text
def preprocess_text(text):
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower() 
    text = ' '.join([word for word in text.split() if word not in english_stopwords]) 
    return text

# API route for message classification
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    preprocessed_message = preprocess_text(message)
    message_vec = vectorizer.transform([preprocessed_message])
    prediction = model.predict(message_vec)[0]
    
    return jsonify({
        'message': message,
        'prediction': 'Spam' if prediction == 1 else 'Ham'
    })

if __name__ == '__main__':
    app.run(debug=True)

