# Streamlit app for clothing review sentiment prediction

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

# Load the saved model
model = tf.keras.models.load_model('clothing_LSTM_model.h5')

# Set up Porter Stemmer and Stopwords
ps = PorterStemmer()
nltk.download('stopwords')

# Function for text preprocessing
def preprocess_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

# Function to convert text to padded sequences
def convert_to_sequence(text, vocab_size=10000, max_len=20):
    one_hot_repr = [tf.keras.preprocessing.text.one_hot(text, vocab_size)]
    padded_seq = pad_sequences(one_hot_repr, maxlen=max_len, padding='pre')
    return padded_seq

# Streamlit UI setup
st.set_page_config(page_title="Clothing Review Sentiment", layout="centered", page_icon="👗")
st.markdown("<h1 style='text-align: center; color: #ff6347;'>Clothing Review Sentiment Predictor</h1>", unsafe_allow_html=True)

# Input from user
review = st.text_area("Enter the clothing review", height=150)

if st.button('Predict'):
    if review:
        # Preprocess the input review
        clean_review = preprocess_review(review)
        seq_input = convert_to_sequence(clean_review)
        
        # Prediction
        prediction = model.predict(seq_input)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"

        # Display result
        st.markdown(f"<h2 style='text-align: center; color: #ff6347;'>The review is {sentiment}</h2>", unsafe_allow_html=True)
        
        # Display stars based on the prediction
        if sentiment == "Positive":
            st.markdown("<h3 style='text-align: center;'>⭐⭐⭐⭐⭐</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='text-align: center;'>⭐</h3>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a review to predict.")
