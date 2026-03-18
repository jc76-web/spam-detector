

"""
Email spam checker
author jon
date february 2026

An AI powered spam detector using machine learnig.
Trained on an sms spam collection dataset from kaggle (5,572 real messages).

Features:
-Visual interface built with streamlit
-Uses a Naive Bayes model to classify emails as spam or not spam
-Provides confidence levels for predictions
Real-time feedback on which words triggered the spam detection"""




# ===========imports===========
import streamlit as st # web interface
import pandas as pd # data handling
from sklearn.feature_extraction.text import TfidfVectorizer #converts text to numbers
from sklearn.naive_bayes import MultinomialNB # machine learning model for classification


# ========= load and prepare data =============
# loads the sms spam collection from kaggle
# using latin1 encoding to handle the special characters in the dataset
data = pd.read_csv("spam.csv", encoding='latin1')

# extract message text and the labels from dataset
messages = data['v2'] # column v2 contains the text messages
labels = data['v1'].map({'spam': 1, 'ham': 0}) # column v1 contains the labels, we map 'spam' to 1 and 'ham' (not spam) to 0 for the model

# ========== train the model =============
# convert the text messages to numbers using TF-IDF vectorizer
# TF-IDF identifies which words are most important in each messages
# TF-IDF measures how many times a term appears in a message (term frequency) and how common that term is across all messages (inverse document frequency)
# this helps the model understand which words are more likely to indicate spam
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(messages)

# this function checks which bad words are in the email text and returns a list of those words
def get_bad_words(email_text, vectorizer, model):
    """find which bad words that triggered spam detection
    
    args:
    email_text: this is the email message to check
    vectorizer: trained the TF-IDF vectorizer used to convert text to numbers
    model: trained the naive Bayes model to predict spam or not spam
    
    returns:
    A list of suspicious words found in the email that are common indicators of spam."""

    # the common spam indicator words to check for

    spam_indicators = ['free', 'win', 'winner', 'offer', 'deal','£', 'call', 'claim', '$', 'congratulations', ' deal', 'limited', 'urgent', 'click', 'offer', 'money', 'urgent', 'prize', 'winner', 'cash', 'credit', 'loan']


    # find which spam words are in this email
    found_bad_words = [word for word in spam_indicators if word in email_text.lower()]

    
    return found_bad_words

# train a naive bayes model on the vectorized messages
# Naive Bayes is a simple and effective machine learning algorithm for text classification tasks like spam detection. It works by calculating the probability of a message being spam based on the presence of certain words and features in the text. The model learns from the training data to identify patterns and associations between words and the likelihood of being spam, allowing it to make predictions on new, unseen messages.
model = MultinomialNB()
model.fit(x, labels)

# ==========build streamlit web interface =============
st.title("email spam checker")
st.write("paste an email below to check if its spam!")

#example test buttons
st.subheader("enter test examples")

col1,col2 = st.columns(2)

with col1:
    if st.button("try example spam email"):
        example_spam = "Congratulations! You've won a free iPhone. Click here to claim your prize now!"
        st.session_state.example_text = example_spam
with col2:
    if st.button("try example safe email"):
        example_safe = "Hey, just wanted to check in and see how you're doing. Let's catch up soon!"
        st.session_state.example_text = example_safe 

# populate the text area with the example email if the user clicked one of the example buttons
default_text = ""
if 'example_text' in st.session_state:
    default_text = st.session_state.example_text


# text area for user to input email text
email_input = st.text_area("enter email text here", value=default_text, height=200)
 
# check for spam when the user clicks the button

if st.button("check for spam"):
    if email_input.strip():

        # convert the users email into numbers using the same vectorizer we trained on the dataset
        email_vector = vectorizer.transform([email_input])

        # get the prediction and the confidence level from the model
        prediction = model.predict(email_vector)[0]
        probability = model.predict_proba(email_vector)[0]

        # displays results
        if prediction == 1:
            # spam detected
            st.error("spam detected!")
            # show confidence level using a progress bar 
            st.write("confidence level:") # this adds the label for progress bar
            st.progress(probability[1]) # this is progress bar # probability[1] is already between 0 and 1 
            st.metric("spam probability", f"{probability[1]*100:.1f}%")

            # highlights the suspicious words that triggered the spam detection
            bad_words = get_bad_words(email_input, vectorizer, model) # call the bad_words function
            if bad_words:
                st.warning("suspicious words found: " + ", ".join(bad_words))

        else:
            # safe email
            st.success("this email is not spam.")

            # show confindence with progress bar
            st.write("confidence level:") # this adds the label for progress bar
            st.progress(probability[0]) # this is progress bar
            st.metric("safe probability", f"{probability[0]*100:.1f}%")

        
    else:
        st.warning("please enter some text!")