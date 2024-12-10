import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure SSL and download NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function to generate responses
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Main Streamlit app function
def main():
    st.title("SmartBot: An Interactive Chatbot Powered by NLP")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to SmartBot! Type a message below to start the conversation.")
        
        # Ensure chat log file exists
        chat_log_path = "chat_log.csv"
        if not os.path.exists(chat_log_path):
            with open(chat_log_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("You:")
        if user_input.strip():  # Validate non-empty input
            response = chatbot(user_input)
            st.text_area("SmartBot:", value=response, height=100, max_chars=None)

            # Log the conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(chat_log_path, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with SmartBot. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            with open("chat_log.csv", 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"SmartBot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        except FileNotFoundError:
            st.error("No conversation history found.")

    elif choice == "About":
        st.subheader("About This Project")
        st.write("""
        **SmartBot** is an interactive chatbot built using modern NLP techniques. 
        The chatbot is designed to understand user intents and provide contextually relevant responses.

        ### Key Features:
        - **NLP-Based Intent Recognition**: Uses TF-IDF Vectorization and Logistic Regression to classify user inputs.
        - **Interactive Interface**: Built with Streamlit, offering a clean and responsive UI.
        - **Conversation Logging**: Saves all interactions for review and analysis.

        ### Technologies Used:
        - **Python**: Core programming language.
        - **Natural Language Toolkit (NLTK)**: For text tokenization.
        - **Scikit-learn**: For machine learning model creation.
        - **Streamlit**: For developing the user interface.

        ### Project Goal:
        The goal is to provide an accessible example of combining machine learning with web application frameworks to create intelligent conversational agents.
        """)

if __name__ == '__main__':
    main()
