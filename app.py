from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the CSV file containing hotel reviews
df = pd.read_csv('deceptive-opinion.csv')

# Load the trained classifier and vectorizer for fake review detection
classifier = joblib.load('models/classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_fake_review', methods=['POST'])
def detect_fake_review():
    review_text = request.form['review_text']
    
    # Preprocess the review text
    review_tfidf = vectorizer.transform([review_text])
    
    # Predict using the trained classifier
    prediction = classifier.predict(review_tfidf)
    
    # Return the prediction
    if prediction[0] == 'truthful':
        result = 'This review is truthful.'
    else:
        result = 'This review is deceptive.'
    
    return render_template('index.html', result=result)

@app.route('/hotel_reviews')
def hotel_reviews():
    # Get unique hotel names and sort them
    hotels = df['hotel'].unique()
    hotels.sort()
    
    # Create a dictionary to hold hotel info and reviews, with web-hosted images
    hotel_info_dict = {
        hotel: {
            "image": f"https://example.com/images/hotels/{hotel.replace(' ', '_').lower()}.jpg",
            "reviews": df[df['hotel'] == hotel].to_dict('records')
        } 
        for hotel in hotels
    }
    
    # Pass the necessary filters to the template
    filters = {
        'deceptive': ['truthful', 'deceptive'],
        'polarity': ['positive', 'negative'],
        'source': ['TripAdvisor','Web','MTurk']
    }
    
    return render_template('hotel_reviews.html', hotel_info_dict=hotel_info_dict, filters=filters)

if __name__ == '__main__':
    app.run(debug=True)
