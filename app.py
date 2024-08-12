import streamlit as st
import pickle
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords")
nltk.download("punkt")

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    print(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    print(y)
    return " ".join(y)


tfidf = pickle.load(open("vectorizer17.pkl", "rb"))
model = pickle.load(open("model17.pkl", "rb"))

tfidf = joblib.load("vectorizer17.pkl")
model = joblib.load("model17.pkl")

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    # transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([input_sms])
    print(vector_input)
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
