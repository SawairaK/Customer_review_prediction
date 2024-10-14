# Import necessary libraries
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the Attention Layer class
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        score = self.dense(inputs)
        attention_weights = tf.keras.layers.Softmax(axis=1)(score)
        context_vector = inputs * attention_weights
        context_vector = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
        return context_vector

def load_model_and_tokenizer():
    # Load the trained model
    try:
        model_path = 'clothing_model.h5'  # Update this path if necessary
        loaded_model = tf.keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
        
        # Load the tokenizer
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        return loaded_model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Load model and tokenizer
loaded_model, tokenizer = load_model_and_tokenizer()

# Ensure model and tokenizer are loaded successfully
if loaded_model is None or tokenizer is None:
    st.stop()

# Define maximum sequence length (use the value used during training)
max_seq_length = 62  # Update this value if it is different in your training code

# Preprocessing function for user input
def preprocess_review(review_text, tokenizer, max_seq_length):
    # Normalize the text
    review_text = review_text.lower()
    review_text = re.sub(r'[^\w\s]', '', review_text)  # Remove punctuation
    
    # Convert the text to sequences
    sequence = tokenizer.texts_to_sequences([review_text])
    
    # Pad the sequence to match the model's expected input shape
    padded_sequence = pad_sequences(sequence, maxlen=max_seq_length, padding='post')
    
    return padded_sequence

# Streamlit app layout with styling
st.set_page_config(page_title="Clothing Review Sentiment Analysis", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Clothing Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B0082;'>Created by: Sawaira Waheed</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #FFD700;'>⭐⭐⭐⭐⭐</h2>", unsafe_allow_html=True)

st.write("### Please enter your clothing review below:")

# Input for the review
review_text = st.text_area("Review Text", height=150, placeholder="Type your review here...")

# Button to predict sentiment
if st.button("Predict"):
    if review_text.strip():
        # Preprocess the input review
        padded_review = preprocess_review(review_text, tokenizer, max_seq_length)
        
        # Make a prediction
        prediction = loaded_model.predict(padded_review)
        
        # Convert prediction to binary output (0 or 1)
        predicted_label = (prediction[0][0] > 0.5).astype(int)
        
        # Display the prediction result with stars
        if predicted_label == 1:
            st.markdown("<div style='text-align: center; color: green; font-size: 1.5em;'>✅ The review is positive!</div>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #FFD700;'>⭐⭐⭐⭐⭐</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; color: red; font-size: 1.5em;'>❌ The review is negative.</div>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #FFD700;'>⭐</h2>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a review text before predicting.")

# Footer with additional info
st.markdown("<h4 style='text-align: center; color: #808080;'>Model created using RNN with GRU and Attention Layer</h4>", unsafe_allow_html=True)
