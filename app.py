import streamlit as st
import pickle

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

tfidf = pickle.load(open('vectorizer.pkl','rb'))

model = pickle.load(open('model.pkl','rb'))

st.title("Email/sms Spam Classification")

ps = PorterStemmer()

def text_process(mess):
    mess = re.sub('[^a-zA-Z]',' ',mess)
    mess = mess.lower()
    
    mess = mess.split()
    
    mess = [ps.stem(words) for words in mess if words not in stopwords.words("english")]
    
    mess = ' '.join(mess)
    
    return mess

input_sms = st.text_input("Enter the message")
if st.button('Predict'):

    transform_sms = text_process(input_sms)

    vector_input = tfidf.transform([transform_sms])

    result = model.predict(vector_input)[0]

    if result==1:
        st.header("Spam")
    else :
        st.header("ham")
