import streamlit as st
import time
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
import re
import string
import speech_recognition as sr

nltk.download('punkt')

# Load the model
model = pickle.load(open('pickle/bestmodel.pkl', 'rb'))
vectorizer = pickle.load(open('pickle/TFIDFvectorizer.pkl', 'rb'))

# Load the dataset used for training
import pandas as pd
dataset = pd.read_csv('./Dataset/text-data.csv')

# Extract offensive words from the dataset
offensive_words_dataset = set()

for text in dataset['text']:
    words = nltk.word_tokenize(text)
    for word in words:
        offensive_words_dataset.add(word.lower())

additional_offensive_words = []

# Combine the additional offensive words with the ones from the dataset
all_offensive_words = additional_offensive_words + list(offensive_words_dataset)

# Initialize NLTK's Porter Stemmer
ps = PorterStemmer()

# Function to preprocess input text
def preprocess_text(text):
    return text

# Function to make predictions
def predict(text):
    preprocessed_text = preprocess_text(text)
    
    # Check if the entire preprocessed text contains at least one bad keyword
    if any(word in preprocessed_text.lower() for word in all_offensive_words):
        return 1  # If any offensive word is found, classify as cyberbullying

    # Vectorize the preprocessed text
    vectorized_text = vectorizer.transform([preprocessed_text])
    
    # Make prediction using the model
    prediction = model.predict(vectorized_text)[0]
    
    return prediction

# Function to transcribe audio file to text
def transcribe_audio(audio_path):
    try:
        # Convert audio to text using speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            audio_text = recognizer.recognize_google(audio_data)
        
        return audio_text
    except Exception as e:
        st.error(f"Error occurred while transcribing audio: {str(e)}")
        return None

def main():
    st.title("Audio Cyberbullying Detection")

    # Audio file upload
    audio_file = st.file_uploader("Upload an audio file", type=["wav"])

    if audio_file is not None:
        try:
            # Transcribe audio to text
            audio_text = transcribe_audio(audio_file)
    
            if audio_text is not None:
                st.write("Transcribed Text:")
                st.write(audio_text)
    
                # Make prediction
                prediction = predict(audio_text)
    
    
                st.write(" ")
                st.write("Posted Message:")

                if prediction == 1:
                    if audio_text.strip() != "":
                        st.markdown("""
                    <style>
                    @keyframes blink {
                        0% { opacity: 1; }
                        50% { opacity: 0; }
                        100% { opacity: 1; }
                    }
                    .red-overlay {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100vw;
                        height: 100vh;
                        background-color: rgba(255, 0, 0, 0.5);
                        z-index: 99999;
                        animation: blink 1s infinite;
                    }
                    </style>
                    <div class="red-overlay"></div>
                    """, unsafe_allow_html=True)
                    
                        time.sleep(2) 
                        st.markdown("""<style>.red-overlay { display: none; }</style>""", unsafe_allow_html=True)
                        st.error("The audio file was deleted by the system due to inappropriate content.")
                    else:
                        st.error("Error occurred while transcribing audio.")
                else:
                    st.audio(audio_file, format='audio/wav')  
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
