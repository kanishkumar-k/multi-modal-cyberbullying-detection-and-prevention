import streamlit as st
import time
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string
import transformers
import torch
from hugchat import hugchat
from hugchat.login import Login
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('punkt')

#nltk.download('vader_lexicon')
#sia = SentimentIntensityAnalyzer()
# Load the model
model = pickle.load(open('pickle/bestmodel.pkl', 'rb'))
vectorizer = pickle.load(open('pickle/TFIDFvectorizer.pkl', 'rb'))

#model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
#tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
#model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
# Load the dataset used for training


#model_name = "unitary/toxic-bert"
#tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
#model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

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
    # Replace special characters with regex-friendly patterns
    text = re.sub(r'[@#$*]', r'[aeiou]', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    return text.lower()  # Convert text to lowercase

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

default_prompt = "provide me a cyberbullying severity score on a scale of 0 to 10 in whole numbers based on its context for the below input for the input prompt"
email="satyavarsshenira.24cs@licet.ac.in"
passwd="302118@Sa"

#def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
 #   sign = Login(email, passwd)
  #  cookies = sign.login()
    # Create ChatBot                        
   # chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    #return chatbot.chat(prompt_input)

def predict_severity(text):
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot                        
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(default_prompt+text)
    #preprocessed_text = preprocess_text(text)
    
    # Tokenize the text
    #inputs = tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True)
    
    # Make prediction using the model
    #outputs = model(**inputs)
    #logits = outputs.logits
    #predicted_class = torch.argmax(logits, dim=1).item()
    
    # Scale the predicted class to the severity range (0 to 10)
    #severity_level = (predicted_class / 4) * 10
    
    #return severity_level

# Main Streamlit app
def main():
    st.title("Severity Determination")

    # Input text area for user input
    input_text = st.text_area("Enter the text to analyze")

    # Check if input text contains special characters
    contains_special_chars = re.search(r'[@#$*]', input_text) is not None

    # Consent checkbox only if input text contains special characters
    if contains_special_chars:
        consent = st.checkbox("I'm aware masking derogatory words may lead to account suspension if found.")

    # Initialize list to store all comments and warnings
    if "comments" not in st.session_state:
        st.session_state.comments = []

    if "warnings_shown" not in st.session_state:
        st.session_state.warnings_shown = set()

    # Predict button 
    if st.button("Submit"):
        if input_text.strip() == "":
            st.error("Please provide some text!")
        elif contains_special_chars and not consent:
            st.warning("Please acknowledge that you are not using any masked derogatory words.")
        else:
            # Make prediction
            prediction = predict(input_text) 
            severity_level=predict_severity(input_text)

            # Display prediction result and store the comment
            if prediction == 1:
                if input_text not in st.session_state.warnings_shown:
                    st.session_state.warnings_shown.add(input_text)
                    st.warning(f"Warning: It may contain inappropriate words.")
                    st.warning("severity level")
                    st.warning(severity_level)
                    # Play system alert sound (beep)
                    # Apply red tint to entire screen with blink effect
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
                    """, 
                    unsafe_allow_html=True)
                    
                    time.sleep(2)  # Wait for 2 seconds
                    st.markdown("""<style>.red-overlay { display: none; }</style>""", unsafe_allow_html=True)
                else:
                    st.session_state.comments.append(("The message was deleted by system due to inappropriate content", True))  # Flag as deleted
            else:
                st.session_state.comments.append((input_text, False))

    # Display all comments
    st.header("Posted Messages")
    for comment, deleted in st.session_state.comments:
        if deleted:
            st.error(comment)
        else:
            st.write(comment)
    
    # Show consent checkbox only if input text contains special characters
    if contains_special_chars:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.subheader("Masking Consent")
        if consent:
            st.success("Thank you for acknowledging!")
    
    st.subheader("Model Accuracy")
    expander_accuracy = st.expander("Information", expanded=False)
    with expander_accuracy:
        st.info("Model Accuracy using Random Forest (RF) Classifier!")
        st.warning("Accuracy:  **_91.70 %_**")

if __name__ == "__main__":
    main()
